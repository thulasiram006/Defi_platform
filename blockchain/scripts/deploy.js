import fs from "fs";
import path from "path";
import { network } from "hardhat";

async function main() {
  const { ethers } = await network.connect();

  console.log("🚀 Deploying Lending smart contract...");
  const Lending = await ethers.getContractFactory("Lending");
  const lending = await Lending.deploy();
  await lending.waitForDeployment();

  const contractAddress = await lending.getAddress();
  console.log(`✅ Lending deployed successfully to: ${contractAddress}`);

  // -------------------------------------------------------------
  // 1. AUTOMATICALLY UPDATE BACKEND .env FILE
  // -------------------------------------------------------------
  const envPath = path.resolve("../backend/.env");
  const envContent = `PORT=5000\nRPC_URL=http://127.0.0.1:8545\nCONTRACT_ADDRESS=${contractAddress}\nPRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80\nDB_HOST=localhost\nDB_USER=root\nDB_PASSWORD=root\nDB_NAME=defilens\n`;

  fs.writeFileSync(envPath, envContent);
  console.log(`📝 Updated backend/.env with CONTRACT_ADDRESS=${contractAddress}`);

  // -------------------------------------------------------------
  // 2. AUTOMATICALLY UPDATE checkLoan.js FILE
  // -------------------------------------------------------------
  const checkLoanPath = path.resolve("scripts/checkLoan.js");
  const checkLoanContent = `import { network } from "hardhat";

const CONTRACT_ADDRESS = "${contractAddress}";

async function main() {
  const { ethers } = await network.connect();
  const lending = await ethers.getContractAt("Lending", CONTRACT_ADDRESS);

  console.log("\\n================================");
  console.log("⛓️ FETCHING ALL ON-CHAIN LOANS");
  console.log("================ lash\\n");

  const allLoans = await lending.getAllLoans();
  console.log("Total Loans Count:", allLoans.length);

  allLoans.forEach((loan, index) => {
    console.log(\`\\n--- Loan #\${index} ---\`);
    console.log("Borrower:", loan.borrower);
    console.log("Principal:", ethers.formatEther(loan.principal), "ETH");
    console.log("Collateral:", ethers.formatEther(loan.collateral), "ETH");
    console.log("Active:", loan.active);
  });
  console.log("\\n================ lash\\n");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
`;

  fs.writeFileSync(checkLoanPath, checkLoanContent);
  console.log(`📝 Updated scripts/checkLoan.js with CONTRACT_ADDRESS="${contractAddress}"`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});