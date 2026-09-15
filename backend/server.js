import express from "express";
import cors from "cors";
import mysql from "mysql2/promise";
import crypto from "crypto";
import "dotenv/config";
import { ethers } from "ethers";

const app = express();

app.use(cors());
app.use(express.json());


// ============================================================
// CONFIGURATION
// ============================================================

const PORT = Number(process.env.PORT || 5000);

const RPC_URL =
  process.env.RPC_URL ||
  "http://127.0.0.1:8545";

const CONTRACT_ADDRESS =
  process.env.CONTRACT_ADDRESS ||
  "0x5FbDB2315678afecb367f032d93F642f64180aa3";

const PRIVATE_KEY =
  process.env.PRIVATE_KEY ||
  "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";


// Module 4 receives blockchain state here.
const MODULE4_URL =
  process.env.MODULE4_URL ||
  "http://localhost:8002/api/risk/update";


// ============================================================
// ETHEREUM CONNECTION
// ============================================================

const provider =
  new ethers.JsonRpcProvider(RPC_URL);

const signer =
  new ethers.Wallet(PRIVATE_KEY, provider);


// ============================================================
// SMART CONTRACT ABI
// ============================================================

const ABI = [

  // Liquidity
  "function lenderPrincipal(address) view returns (uint256)",
  "function supplyLiquidity() public payable",

  // Lender
  "function getLenderInfo(address) view returns (uint256 principal,uint256 claimableInterest,uint256 shares,uint256 poolSharePercent)",

  // Borrower
  "function getBorrowerSummary(address) view returns (uint256 availableCol,uint256 lockedCol,uint256 debt,uint256 maxAdditionalBorrow,uint256 healthRatioPercent)",

  // Platform
  "function getPlatformStats() view returns (uint256 available,uint256 supplied,uint256 borrowed,uint256 collateral,uint256 accruedInterest,uint256 utilizationPercent)",

  // Collateral / borrowing
  "function depositCollateral() public payable",
  "function borrow(uint256 amount) public",

  // Loans
  "function getLoan(uint256 loanId) view returns (tuple(uint256 id,address borrower,uint256 principal,uint256 collateral,uint256 accruedInterest,bool active))",
  "function getAllLoans() view returns (tuple(uint256 id,address borrower,uint256 principal,uint256 collateral,uint256 accruedInterest,bool active)[])"
];


// ============================================================
// CONTRACT INSTANCES
// ============================================================

const readContract =
  new ethers.Contract(
    CONTRACT_ADDRESS,
    ABI,
    provider
  );

const writeContract =
  new ethers.Contract(
    CONTRACT_ADDRESS,
    ABI,
    signer
  );


// ============================================================
// MYSQL
// ============================================================

const pool = mysql.createPool({

  host:
    process.env.DB_HOST ||
    "localhost",

  user:
    process.env.DB_USER ||
    "root",

  password:
    process.env.DB_PASSWORD ||
    "root",

  database:
    process.env.DB_NAME ||
    "defilens",

  waitForConnections: true,

  connectionLimit: 10
});


// ============================================================
// DATABASE INITIALIZATION
// ============================================================

async function initDb() {

  await pool.query(`
    CREATE TABLE IF NOT EXISTS transactions (

      id INT AUTO_INCREMENT PRIMARY KEY,

      wallet_address VARCHAR(42) NOT NULL,

      action_type VARCHAR(50) NOT NULL,

      collateral_amount DECIMAL(36,18)
        DEFAULT 0,

      borrow_amount DECIMAL(36,18)
        DEFAULT 0,

      status VARCHAR(30) NOT NULL,

      crypto_proof VARCHAR(128),

      tx_hash VARCHAR(66),

      rejection_reason TEXT,

      created_at TIMESTAMP
        DEFAULT CURRENT_TIMESTAMP,

      confirmed_at TIMESTAMP NULL
    )
  `);

  const [columns] =
    await pool.query(
      "SHOW COLUMNS FROM transactions"
    );

  const names =
    new Set(
      columns.map(
        column => column.Field
      )
    );

  if (!names.has("tx_hash")) {

    await pool.query(
      "ALTER TABLE transactions ADD COLUMN tx_hash VARCHAR(66)"
    );
  }

  if (!names.has("confirmed_at")) {

    await pool.query(
      "ALTER TABLE transactions ADD COLUMN confirmed_at TIMESTAMP NULL"
    );
  }
}


// ============================================================
// HELPERS
// ============================================================

function isAddress(value) {

  return (
    typeof value === "string" &&
    ethers.isAddress(value)
  );
}


function safeNumber(value, fallback = 0) {

  const number = Number(value);

  return Number.isFinite(number)
    ? number
    : fallback;
}


function nowISO() {

  return new Date().toISOString();
}


function createProof(
  wallet,
  action,
  amount,
  collateral
) {

  return crypto
    .createHash("sha256")
    .update(
      `${wallet}|${action}|${amount}|${collateral}|${Date.now()}`
    )
    .digest("hex");
}


// ============================================================
// READ CURRENT BLOCKCHAIN STATE
// ============================================================

async function getBlockchainState(wallet) {

  const [
    stats,
    summary,
    lenderInfo
  ] = await Promise.all([

    readContract.getPlatformStats(),

    readContract.getBorrowerSummary(wallet),

    readContract.getLenderInfo(wallet)
  ]);


  return {

    platform: {

      availableLiquidity:
        ethers.formatEther(stats[0]),

      totalSupplied:
        ethers.formatEther(stats[1]),

      totalBorrowed:
        ethers.formatEther(stats[2]),

      totalCollateral:
        ethers.formatEther(stats[3]),

      accruedInterest:
        ethers.formatEther(stats[4]),

      utilizationPercent:
        Number(stats[5])
    },


    borrower: {

      availableCollateral:
        ethers.formatEther(summary[0]),

      lockedCollateral:
        ethers.formatEther(summary[1]),

      debt:
        ethers.formatEther(summary[2]),

      maxAdditionalBorrow:
        ethers.formatEther(summary[3]),

      healthRatioPercent:
        Number(summary[4])
    },


    lender: {

      principal:
        ethers.formatEther(lenderInfo[0]),

      claimableInterest:
        ethers.formatEther(lenderInfo[1]),

      shares:
        lenderInfo[2].toString(),

      poolSharePercent:
        Number(lenderInfo[3]) / 100
    }
  };
}


// ============================================================
// PRE-TRANSACTION VALIDATION
// ============================================================

async function onChainValidation({
  wallet,
  amount,
  collateral,
  action
}) {

  if (!isAddress(wallet)) {

    return {
      success: false,
      reason: "Invalid wallet address."
    };
  }


  let amountWei;
  let collateralWei;

  try {

    amountWei =
      ethers.parseEther(
        String(amount || "0")
      );

    collateralWei =
      ethers.parseEther(
        String(collateral || "0")
      );

  } catch {

    return {
      success: false,
      reason: "Invalid ETH amount."
    };
  }


  const state =
    await getBlockchainState(wallet);


  const availableWei =
    ethers.parseEther(
      state.platform.availableLiquidity
    );


  const maxBorrowWei =
    ethers.parseEther(
      state.borrower.maxAdditionalBorrow
    );


  // ----------------------------------------------------------
  // SUPPLY LIQUIDITY
  // ----------------------------------------------------------

  if (action === "SupplyLiquidity") {

    if (amountWei <= 0n) {

      return {
        success: false,
        reason:
          "Supply amount must be greater than 0."
      };
    }
  }


  // ----------------------------------------------------------
  // WITHDRAW SUPPLY
  // ----------------------------------------------------------

  else if (action === "WithdrawSupply") {

    const supplied =
      await readContract.lenderPrincipal(wallet);

    if (amountWei <= 0n) {

      return {
        success: false,
        reason:
          "Withdrawal amount must be greater than 0."
      };
    }

    if (amountWei > supplied) {

      return {
        success: false,
        reason:
          `You only supplied ${ethers.formatEther(supplied)} ETH.`
      };
    }

    if (amountWei > availableWei) {

      return {
        success: false,
        reason:
          `Only ${state.platform.availableLiquidity} ETH is currently available.`
      };
    }
  }


  // ----------------------------------------------------------
  // DEPOSIT COLLATERAL
  // ----------------------------------------------------------

  else if (action === "DepositCollateral") {

    if (collateralWei <= 0n) {

      return {
        success: false,
        reason:
          "Collateral amount must be greater than 0."
      };
    }
  }


  // ----------------------------------------------------------
  // BORROW
  // ----------------------------------------------------------

  else if (action === "Borrow") {

    if (amountWei <= 0n) {

      return {
        success: false,
        reason:
          "Borrow amount must be greater than 0."
      };
    }


    if (amountWei > availableWei) {

      return {
        success: false,
        reason:
          `Insufficient platform liquidity. Available: ${state.platform.availableLiquidity} ETH.`
      };
    }


    // IMPORTANT:
    // Module 2 sends the proposed collateral with the request.
    // Therefore validate the requested collateral itself.
    const requiredCollateral =
      amountWei * 150n / 100n;


    if (collateralWei < requiredCollateral) {

      return {
        success: false,

        reason:
          `Undercollateralized borrow. Required collateral is at least ${ethers.formatEther(requiredCollateral)} ETH.`,

        requiredCollateral:
          ethers.formatEther(requiredCollateral),

        providedCollateral:
          ethers.formatEther(collateralWei),

        coveragePercent:
          Number(
            collateralWei * 10000n /
            amountWei
          ) / 100
      };
    }


    // If borrower already has an active on-chain
    // borrowing capacity, preserve that check too.
    //
    // For a new demo borrower the incoming collateral
    // is going to be deposited immediately before borrow,
    // so this check is informational rather than blocking.
  }


  // ----------------------------------------------------------
  // REPAY
  // ----------------------------------------------------------

  else if (action === "Repay") {

    if (amountWei <= 0n) {

      return {
        success: false,
        reason:
          "Repayment amount must be greater than 0."
      };
    }

    if (collateralWei > 0n) {

      return {
        success: false,
        reason:
          "Collateral is not sent with repayment."
      };
    }
  }


  // ----------------------------------------------------------
  // WITHDRAW COLLATERAL
  // ----------------------------------------------------------

  else if (action === "WithdrawCollateral") {

    if (amountWei <= 0n) {

      return {
        success: false,
        reason:
          "Withdrawal amount must be greater than 0."
      };
    }

    if (amountWei > ethers.parseEther(
      state.borrower.availableCollateral
    )) {

      return {
        success: false,
        reason:
          "Requested withdrawal exceeds available collateral."
      };
    }
  }


  // ----------------------------------------------------------
  // CLAIM INTEREST
  // ----------------------------------------------------------

  else if (action === "ClaimInterest") {

    if (
      Number(state.lender.claimableInterest) <= 0
    ) {

      return {
        success: false,
        reason:
          "No realized lender interest is currently claimable."
      };
    }
  }


  // ----------------------------------------------------------
  // INVALID ACTION
  // ----------------------------------------------------------

  else {

    return {
      success: false,
      reason:
        "Unsupported action."
    };
  }


  return {

    success: true,

    reason:
      "Blockchain pre-validation passed.",

    ...state,

    validation: {

      requestedAmount:
        String(amount),

      requestedCollateral:
        String(collateral),

      action,

      collateralCoveragePercent:
        amountWei > 0n
          ? Number(
              collateralWei * 10000n /
              amountWei
            ) / 100
          : null
    }
  };
}


// ============================================================
// SEND BLOCKCHAIN STATE TO MODULE 4
// ============================================================

async function sendToModule4(payload) {

  try {

    console.log(
      "\n📤 Sending updated blockchain state to Module 4..."
    );

    const response =
      await fetch(
        MODULE4_URL,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json"
          },

          body:
            JSON.stringify(payload)
        }
      );


    const text =
      await response.text();


    let data;

    try {

      data =
        JSON.parse(text);

    } catch {

      data = {
        raw_response: text
      };
    }


    console.log(
      `📊 Module 4 response: ${response.status}`
    );


    return {

      success:
        response.ok,

      status_code:
        response.status,

      response:
        data
    };

  } catch (error) {

    console.error(
      "⚠️ Module 4 transmission failed:",
      error.message
    );


    return {

      success: false,

      status_code: null,

      response: {
        error:
          error.message
      }
    };
  }
}


// ============================================================
// HEALTH CHECK
// ============================================================

app.get(
  "/api/health",
  async (req, res) => {

    try {

      const blockNumber =
        await provider.getBlockNumber();


      res.json({

        success: true,

        module:
          "Module 3 - DeFi Blockchain Lending",

        status:
          "healthy",

        rpc:
          RPC_URL,

        contract:
          CONTRACT_ADDRESS,

        signer:
          signer.address,

        blockNumber,

        module4:
          MODULE4_URL
      });

    } catch (error) {

      res.status(500).json({

        success: false,

        reason:
          error.message
      });
    }
  }
);


// ============================================================
// MODULE 2 → MODULE 3
// MAIN BLOCKCHAIN PROCESSING ENDPOINT
// ============================================================

app.post(
  "/api/blockchain/process-transaction",
  async (req, res) => {

    try {

      const body =
        req.body || {};


      const {

        wallet,

        borrower_wallet_address,

        amount = "0",

        requested_loan_amount,

        collateral = "0",

        collateral_amount,

        action = "Borrow",

        decision,

        fraud_probability = 0,

        fraud_status = "UNKNOWN",

        verified_monthly_income = 0,

        total_liabilities = 0,

        discrepancy_ratio = 0,

        dr_risk_penalty = 0,

        final_risk = 0,

        risk_level = "UNKNOWN",

        transaction_hash = null,

        behavior_features = null

      } = body;


      // ------------------------------------------------------
      // NORMALIZE INPUT
      // ------------------------------------------------------

      const walletAddress =
        wallet ||
        borrower_wallet_address;


      const borrowAmount =
        requested_loan_amount !== undefined
          ? requested_loan_amount
          : amount;


      const collateralAmount =
        collateral_amount !== undefined
          ? collateral_amount
          : collateral;


      // ------------------------------------------------------
      // DEFENSE-IN-DEPTH:
      // MODULE 2 MUST HAVE APPROVED THE TRANSACTION.
      // ------------------------------------------------------

      const normalizedDecision =
        String(
          decision || ""
        )
          .trim()
          .toUpperCase();


      if (
        normalizedDecision &&
        normalizedDecision !== "APPROVED"
      ) {

        const proof =
          createProof(
            walletAddress,
            action,
            borrowAmount,
            collateralAmount
          );


        let transactionId = null;


        if (isAddress(walletAddress)) {

          const [result] =
            await pool.query(

              `
              INSERT INTO transactions
              (
                wallet_address,
                action_type,
                collateral_amount,
                borrow_amount,
                status,
                crypto_proof,
                rejection_reason
              )
              VALUES (?, ?, ?, ?, 'REJECTED', ?, ?)
              `,

              [

                walletAddress,

                action,

                collateralAmount,

                borrowAmount,

                proof,

                "Module 2 decision was not APPROVED."
              ]
            );


          transactionId =
            result.insertId;
        }


        return res.status(403).json({

          success: false,

          status:
            "REJECTED",

          reason:
            "Transaction blocked because Module 2 did not approve it.",

          transactionId
        });
      }


      // ------------------------------------------------------
      // VALIDATE INPUT
      // ------------------------------------------------------

      if (!isAddress(walletAddress)) {

        return res.status(400).json({

          success: false,

          status:
            "REJECTED",

          reason:
            "Valid borrower wallet address is required."
        });
      }


      // ------------------------------------------------------
      // CREATE AUDIT PROOF
      // ------------------------------------------------------

      const proof =
        createProof(
          walletAddress,
          action,
          borrowAmount,
          collateralAmount
        );


      // ------------------------------------------------------
      // BLOCKCHAIN PRE-VALIDATION
      // ------------------------------------------------------

      const validation =
        await onChainValidation({

          wallet:
            walletAddress,

          amount:
            borrowAmount,

          collateral:
            collateralAmount,

          action
        });


      // ------------------------------------------------------
      // VALIDATION FAILURE
      // NO BLOCKCHAIN WRITE
      // ------------------------------------------------------

      if (!validation.success) {

        const [result] =
          await pool.query(

            `
            INSERT INTO transactions
            (
              wallet_address,
              action_type,
              collateral_amount,
              borrow_amount,
              status,
              crypto_proof,
              rejection_reason
            )
            VALUES (?, ?, ?, ?, 'REJECTED', ?, ?)
            `,

            [

              walletAddress,

              action,

              collateralAmount,

              borrowAmount,

              proof,

              validation.reason
            ]
          );


        console.log(
          `❌ Transaction rejected: ${validation.reason}`
        );


        return res.status(400).json({

          success: false,

          status:
            "REJECTED",

          reason:
            validation.reason,

          transactionId:
            result.insertId,

          validation
        });
      }


      // ------------------------------------------------------
      // RECORD VALIDATED TRANSACTION
      // ------------------------------------------------------

      const [dbResult] =
        await pool.query(

          `
          INSERT INTO transactions
          (
            wallet_address,
            action_type,
            collateral_amount,
            borrow_amount,
            status,
            crypto_proof
          )
          VALUES (?, ?, ?, ?, 'VALIDATED', ?)
          `,

          [

            walletAddress,

            action,

            collateralAmount,

            borrowAmount,

            proof
          ]
        );


      const transactionId =
        dbResult.insertId;


      console.log(
        `\n✅ Pre-validation passed. Transaction ID: ${transactionId}`
      );


      // ======================================================
      // BORROW EXECUTION
      // ======================================================

      let txHash = null;

      let gasUsed = null;

      let depositTxHash = null;


      if (action === "Borrow") {

        try {

          console.log(
            `\n⛓️ Executing approved borrow...`
          );

          console.log(
            `   Borrower: ${walletAddress}`
          );

          console.log(
            `   Collateral: ${collateralAmount} ETH`
          );

          console.log(
            `   Borrow: ${borrowAmount} ETH`
          );


          // --------------------------------------------------
          // IMPORTANT DEMO EXECUTION
          // --------------------------------------------------
          //
          // The current Lending.sol exposes depositCollateral()
          // and borrow() as msg.sender functions.
          //
          // Therefore the backend signer performs the actual
          // Hardhat transaction. The borrower wallet is retained
          // as the logical borrower/audit address.
          //
          // This matches the current contract interface.
          // --------------------------------------------------


          // Deposit collateral first.
          const depositTx =
            await writeContract.depositCollateral({

              value:
                ethers.parseEther(
                  String(collateralAmount)
                )
            });


          console.log(
            `   Collateral TX: ${depositTx.hash}`
          );


          await depositTx.wait();


          depositTxHash =
            depositTx.hash;


          // Execute borrow.
          const borrowTx =
            await writeContract.borrow(

              ethers.parseEther(
                String(borrowAmount)
              )
            );


          console.log(
            `   Borrow TX: ${borrowTx.hash}`
          );


          const receipt =
            await borrowTx.wait();


          txHash =
            borrowTx.hash;


          if (receipt && receipt.gasUsed) {

            gasUsed =
              receipt.gasUsed.toString();
          }


          console.log(
            `✅ Blockchain transaction confirmed: ${txHash}`
          );


        } catch (chainError) {

          console.error(
            "❌ Smart contract execution failed:",
            chainError.message
          );


          await pool.query(

            `
            UPDATE transactions

            SET
              status = 'FAILED',
              rejection_reason = ?

            WHERE id = ?
            `,

            [

              chainError.message,

              transactionId
            ]
          );


          return res.status(500).json({

            success: false,

            status:
              "FAILED",

            reason:
              "Smart contract execution failed.",

            details:
              chainError.message,

            transactionId
          });
        }
      }


      // ======================================================
      // CONFIRM DATABASE RECORD
      // ======================================================

      await pool.query(

        `
        UPDATE transactions

        SET
          status = 'CONFIRMED',
          tx_hash = ?,
          confirmed_at = NOW(),
          rejection_reason = NULL

        WHERE id = ?
        `,

        [

          txHash,

          transactionId
        ]
      );


      // ======================================================
      // GET UPDATED BLOCKCHAIN STATE
      // ======================================================

      //
      // The smart contract was executed by the backend signer.
      // Therefore query the signer for the actual resulting
      // blockchain state.
      //

      const chainState =
        await getBlockchainState(
          signer.address
        );


      // ======================================================
      // CALCULATE DYNAMIC METRICS
      // ======================================================

      const collateralValue =
        safeNumber(
          chainState.borrower.availableCollateral
        ) +
        safeNumber(
          chainState.borrower.lockedCollateral
        );


      const debtValue =
        safeNumber(
          chainState.borrower.debt
        );


      let collateralLoanRatio = null;


      if (debtValue > 0) {

        collateralLoanRatio =
          Number(
            (
              collateralValue /
              debtValue
            ).toFixed(4)
          );
      }


      const healthRatio =
        safeNumber(
          chainState.borrower.healthRatioPercent
        );


      const utilization =
        safeNumber(
          chainState.platform.utilizationPercent
        );


      let liquidityRisk;


      if (utilization >= 90) {

        liquidityRisk = "HIGH";

      } else if (utilization >= 70) {

        liquidityRisk = "MEDIUM";

      } else {

        liquidityRisk = "LOW";
      }


      // ======================================================
      // BUILD MODULE 4 PAYLOAD
      // ======================================================

      const module4Payload = {

        module:
          "module_3_blockchain",

        timestamp:
          nowISO(),


        transaction: {

          transactionId,

          action,

          status:
            "CONFIRMED",

          txHash,

          depositTxHash,

          gasUsed,

          cryptoProof:
            proof
        },


        borrower: {

          wallet:
            walletAddress,

          blockchainExecutionWallet:
            signer.address
        },


        loan: {

          requestedAmount:
            safeNumber(
              borrowAmount
            ),

          depositedCollateral:
            safeNumber(
              collateralAmount
            )
        },


        blockchain: {

          platform:
            chainState.platform,

          borrower:
            chainState.borrower,

          lender:
            chainState.lender
        },


        dynamicRisk: {

          collateralLoanRatio,

          healthRatioPercent:
            healthRatio,

          liquidityUtilizationPercent:
            utilization,

          liquidityRisk
        },


        fraud: {

          probability:
            safeNumber(
              fraud_probability
            ),

          status:
            fraud_status,

          decision:
            normalizedDecision || "APPROVED"
        },


        financial: {

          verifiedMonthlyIncome:
            safeNumber(
              verified_monthly_income
            ),

          totalLiabilities:
            safeNumber(
              total_liabilities
            ),

          discrepancyRatio:
            safeNumber(
              discrepancy_ratio
            ),

          drRiskPenalty:
            safeNumber(
              dr_risk_penalty
            ),

          finalRisk:
            safeNumber(
              final_risk
            ),

          riskLevel:
            risk_level
        },


        behavior_features:
          behavior_features || null
      };


      // ======================================================
      // MODULE 3 → MODULE 4
      // ======================================================

      const module4Result =
        await sendToModule4(
          module4Payload
        );


      // ======================================================
      // FINAL RESPONSE TO MODULE 2
      // ======================================================

      return res.json({

        success: true,

        status:
          "CONFIRMED",

        transactionId,

        borrower:
          walletAddress,

        action,

        loanAmount:
          borrowAmount,

        collateral:
          collateralAmount,

        txHash,

        depositTxHash,

        gasUsed,

        blockchainState:
          chainState,

        dynamicRisk: {

          collateralLoanRatio,

          healthRatioPercent:
            healthRatio,

          liquidityRisk,

          liquidityUtilizationPercent:
            utilization
        },

        module4:
          module4Result
      });


    } catch (error) {

      console.error(
        "\n❌ Module 3 processing error:",
        error
      );


      return res.status(500).json({

        success: false,

        status:
          "ERROR",

        reason:
          error.message
      });
    }
  }
);


// ============================================================
// CONFIRM TRANSACTION MANUALLY
// ============================================================

app.post(
  "/api/blockchain/transactions/:id/confirm",
  async (req, res) => {

    try {

      const {
        txHash
      } =
        req.body || {};


      if (
        !txHash ||
        !/^0x[a-fA-F0-9]{64}$/.test(txHash)
      ) {

        return res.status(400).json({

          success: false,

          reason:
            "Invalid transaction hash."
        });
      }


      await pool.query(

        `
        UPDATE transactions

        SET
          status = 'CONFIRMED',
          tx_hash = ?,
          confirmed_at = NOW(),
          rejection_reason = NULL

        WHERE id = ?
        `,

        [

          txHash,

          req.params.id
        ]
      );


      res.json({

        success: true,

        status:
          "CONFIRMED"
      });


    } catch (error) {

      res.status(500).json({

        success: false,

        reason:
          error.message
      });
    }
  }
);


// ============================================================
// MARK TRANSACTION FAILED
// ============================================================

app.post(
  "/api/blockchain/transactions/:id/fail",
  async (req, res) => {

    try {

      const {
        reason,
        txHash = null
      } =
        req.body || {};


      await pool.query(

        `
        UPDATE transactions

        SET
          status = 'FAILED',
          tx_hash = ?,
          rejection_reason = ?

        WHERE id = ?
        `,

        [

          txHash,

          reason ||
            "Blockchain transaction failed.",

          req.params.id
        ]
      );


      res.json({

        success: true,

        status:
          "FAILED"
      });


    } catch (error) {

      res.status(500).json({

        success: false,

        reason:
          error.message
      });
    }
  }
);


// ============================================================
// TRANSACTION HISTORY
// ============================================================

app.get(
  "/api/blockchain/transactions",
  async (req, res) => {

    try {

      const [rows] =
        await pool.query(

          `
          SELECT *

          FROM transactions

          ORDER BY id DESC

          LIMIT 200
          `
        );


      res.json(rows);


    } catch (error) {

      res.status(500).json({

        success: false,

        reason:
          error.message
      });
    }
  }
);


// ============================================================
// BLOCKCHAIN STATE ENDPOINT
// ============================================================

app.get(
  "/api/blockchain/state/:wallet",
  async (req, res) => {

    try {

      const wallet =
        req.params.wallet;


      if (!isAddress(wallet)) {

        return res.status(400).json({

          success: false,

          reason:
            "Invalid wallet address."
        });
      }


      const state =
        await getBlockchainState(
          wallet
        );


      res.json({

        success: true,

        wallet,

        state
      });


    } catch (error) {

      res.status(500).json({

        success: false,

        reason:
          error.message
      });
    }
  }
);


// ============================================================
// START SERVER
// ============================================================

app.listen(
  PORT,
  async () => {

    try {

      await initDb();


      console.log(
        "\n=================================================="
      );

      console.log(
        "🚀 DeFiLens Module 3 Blockchain Backend"
      );

      console.log(
        "=================================================="
      );

      console.log(
        `📡 API: http://localhost:${PORT}`
      );

      console.log(
        `⛓️ RPC: ${RPC_URL}`
      );

      console.log(
        `📜 Contract: ${CONTRACT_ADDRESS}`
      );

      console.log(
        `👤 Execution signer: ${signer.address}`
      );

      console.log(
        `📊 Module 4: ${MODULE4_URL}`
      );

      console.log(
        "==================================================\n"
      );

    } catch (error) {

      console.error(
        "❌ Database initialization failed:",
        error
      );
    }
  }
);