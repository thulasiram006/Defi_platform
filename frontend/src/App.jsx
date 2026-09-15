import React, { useEffect, useState } from "react";
import { ethers } from "ethers";

const API_BASE = "http://localhost:5000/api";
const CONTRACT_ADDRESS = import.meta.env.VITE_CONTRACT_ADDRESS || "0x5FbDB2315678afecb367f032d93F642f64180aa3";

const CONTRACT_ABI = [
  "function supplyLiquidity() external payable",
  "function withdrawSupply(uint256 amount) external",
  "function depositCollateral() external payable",
  "function withdrawAvailableCollateral(uint256 amount) external",
  "function borrow(uint256 amount) external",
  "function repay(uint256 loanId) external payable",
  "function repayFull(uint256 loanId) external payable",
  "function getLoanDebt(uint256 loanId) external view returns (uint256)",
  "function claimInterest() external",
  "function lenderPrincipal(address) external view returns (uint256)",
  "function getLenderInfo(address) external view returns (uint256 principal,uint256 claimableInterest,uint256 shares,uint256 poolSharePercent)",
  "function getBorrowerSummary(address) external view returns (uint256 availableCol,uint256 lockedCol,uint256 debt,uint256 maxAdditionalBorrow,uint256 healthRatioPercent)",
  "function getBorrowerLoans(address) external view returns (tuple(uint256 id,address borrower,uint256 principal,uint256 collateral,uint256 interestRate,uint256 startTime,uint256 lastAccrualTime,uint256 accruedInterest,bool active)[])",
  "function getPlatformStats() external view returns (uint256 available,uint256 supplied,uint256 borrowed,uint256 collateral,uint256 accruedInterest,uint256 utilizationPercent)",
  "function getAvailablePoolLiquidity() external view returns (uint256)"
];

const ACTION_LABELS = {
  SupplyLiquidity: "Supply Liquidity",
  WithdrawSupply: "Withdraw Liquidity",
  DepositCollateral: "Deposit Collateral",
  WithdrawCollateral: "Withdraw Available Collateral",
  Borrow: "Borrow",
  Repay: "Repay Loan",
  ClaimInterest: "Claim Interest"
};

const short = (value) => value ? `${value.slice(0, 6)}...${value.slice(-4)}` : "-";
const dateText = (seconds) => new Date(Number(seconds) * 1000).toLocaleString();

function Card({ title, value, sub, accent = "#38bdf8" }) {
  return (
    <div style={{ background: "#151d2e", padding: 16, borderRadius: 10, border: "1px solid #24324f" }}>
      <div style={{ color: "#94a3b8", fontSize: 13 }}>{title}</div>
      <div style={{ color: accent, fontSize: 24, fontWeight: 700, marginTop: 6 }}>{value}</div>
      {sub && <div style={{ color: "#64748b", fontSize: 12, marginTop: 5 }}>{sub}</div>}
    </div>
  );
}

function Button({ children, onClick, disabled = false, tone = "blue", style = {} }) {
  const backgrounds = { blue: "#38bdf8", green: "#10b981", orange: "#f59e0b", gray: "#475569", red: "#ef4444" };
  return (
    <button disabled={disabled} onClick={onClick} style={{
      background: disabled ? "#334155" : backgrounds[tone],
      color: tone === "blue" || tone === "orange" ? "#07111d" : "#fff",
      border: 0, borderRadius: 6, padding: "10px 16px", fontWeight: 700,
      cursor: disabled ? "not-allowed" : "pointer", ...style
    }}>{children}</button>
  );
}

export default function App() {
  const [activeModule, setActiveModule] = useState("blockchain");
  const [page, setPage] = useState(window.location.hash === "#/lender" ? "lender" : "borrower");
  const [account, setAccount] = useState("");
  const [status, setStatus] = useState("Connect MetaMask to begin.");
  const [intermediateData, setIntermediateData] = useState(null);
  const [dbTransactions, setDbTransactions] = useState([]);
  const [platform, setPlatform] = useState({ available: 0, supplied: 0, borrowed: 0, collateral: 0, accruedInterest: 0, utilization: 0 });
  const [lender, setLender] = useState({ principal: 0, interest: 0, shares: 0, sharePercent: 0 });
  const [borrower, setBorrower] = useState({ availableCollateral: 0, lockedCollateral: 0, debt: 0, maxAdditionalBorrow: 0, healthRatio: 0 });
  const [loans, setLoans] = useState([]);
  const [lastUpdated, setLastUpdated] = useState(null);

  const [supplyInput, setSupplyInput] = useState("");
  const [withdrawInput, setWithdrawInput] = useState("");
  const [collateralInput, setCollateralInput] = useState("");
  const [collateralWithdrawInput, setCollateralWithdrawInput] = useState("");
  const [borrowInput, setBorrowInput] = useState("");
  const [repayInput, setRepayInput] = useState({});

  const setRoute = (next) => {
    window.location.hash = `#/${next}`;
    setPage(next);
  };

  const readContract = async (signerOrProvider) => new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, signerOrProvider);

  const refresh = async (userAddr = account) => {
    if (!window.ethereum || !userAddr) return;
    try {
      // Always create a fresh provider/read so the UI reads the latest mined state.
      const p = new ethers.BrowserProvider(window.ethereum);
      const latestBlock = await p.getBlockNumber();
      const c = await readContract(p);
      const overrides = { blockTag: latestBlock };
      const [stats, currentAvailable, lenderInfo, borrowerInfo, userLoans] = await Promise.all([
        c.getPlatformStats(overrides),
        c.getAvailablePoolLiquidity(overrides),
        c.getLenderInfo(userAddr, overrides),
        c.getBorrowerSummary(userAddr, overrides),
        c.getBorrowerLoans(userAddr, overrides)
      ]);
      setPlatform({
        available: Number(ethers.formatEther(currentAvailable)),
        supplied: Number(ethers.formatEther(stats[1])),
        borrowed: Number(ethers.formatEther(stats[2])),
        collateral: Number(ethers.formatEther(stats[3])),
        accruedInterest: Number(ethers.formatEther(stats[4])),
        utilization: Number(stats[5])
      });
      setLender({
        principal: Number(ethers.formatEther(lenderInfo[0])),
        interest: Number(ethers.formatEther(lenderInfo[1])),
        shares: lenderInfo[2].toString(),
        sharePercent: Number(lenderInfo[3]) / 100
      });
      setBorrower({
        availableCollateral: Number(ethers.formatEther(borrowerInfo[0])),
        lockedCollateral: Number(ethers.formatEther(borrowerInfo[1])),
        debt: Number(ethers.formatEther(borrowerInfo[2])),
        maxAdditionalBorrow: Number(ethers.formatEther(borrowerInfo[3])),
        healthRatio: Number(borrowerInfo[4])
      });
      setLoans(userLoans.map((x) => ({
        id: x.id.toString(), borrower: x.borrower, principal: Number(ethers.formatEther(x.principal)),
        collateral: Number(ethers.formatEther(x.collateral)), rate: Number(x.interestRate),
        startTime: Number(x.startTime), lastAccrualTime: Number(x.lastAccrualTime),
        interest: Number(ethers.formatEther(x.accruedInterest)), active: x.active
      })));

      const res = await fetch(`${API_BASE}/blockchain/transactions`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data)) setDbTransactions(data);
      }
      setLastUpdated(new Date());
    } catch (e) {
      console.error(e);
      setStatus(`Read error: ${e.shortMessage || e.message}`);
    }
  };

  const connectWallet = async () => {
    if (!window.ethereum) {
      setStatus("MetaMask is required.");
      return;
    }
    try {
      await window.ethereum.request({ method: "eth_requestAccounts" });
      const p = new ethers.BrowserProvider(window.ethereum);
      const signer = await p.getSigner();
      const addr = await signer.getAddress();
      setAccount(addr);
      await refresh(addr);
      setStatus(`Connected: ${short(addr)}`);
    } catch (e) {
      setStatus(`Wallet error: ${e.shortMessage || e.message}`);
    }
  };

  useEffect(() => {
    const onHash = () => setPage(window.location.hash === "#/lender" ? "lender" : "borrower");
    window.addEventListener("hashchange", onHash);
    if (window.ethereum) {
      // Reconnect to the already-selected MetaMask account when the page is refreshed.
      window.ethereum.request({ method: "eth_accounts" }).then((accounts) => {
        if (accounts.length) {
          setAccount(accounts[0]);
          refresh(accounts[0]);
        }
      }).catch(() => {});

      window.ethereum.on?.("accountsChanged", (accounts) => {
        if (!accounts.length) setAccount(""); else { setAccount(accounts[0]); refresh(accounts[0]); }
      });
      window.ethereum.on?.("chainChanged", () => window.location.reload());
    }
    if (window.location.hash !== "#/lender" && window.location.hash !== "#/borrower") window.location.hash = "#/borrower";
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // Keep the dashboard synchronized with the blockchain even when another wallet/action
  // changes the pool. This is especially useful when the same borrower takes multiple loans.
  useEffect(() => {
    if (!account) return;
    const interval = setInterval(() => refresh(account), 3000);
    return () => clearInterval(interval);
  }, [account]);

  const repayFull = async (loanId) => {
    if (!account) return setStatus("Please connect your wallet first.");
    let validationId = null;
    try {
      setStatus(`Preparing full repayment for Loan #${loanId}...`);
      const provider = new ethers.BrowserProvider(window.ethereum);
      const readOnlyContract = await readContract(provider);
      const exactDebt = await readOnlyContract.getLoanDebt(loanId);

      if (exactDebt === 0n) {
        setStatus(`Loan #${loanId} has no remaining debt.`);
        await refresh();
        return;
      }

      const validationRes = await fetch(`${API_BASE}/blockchain/process-transaction`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          wallet: account,
          amount: ethers.formatEther(exactDebt),
          collateral: "0",
          action: "Repay"
        })
      });

      const validation = await validationRes.json();
      setIntermediateData(validation);
      validationId = validation.transactionId;

      if (!validationRes.ok || !validation.success) {
        setStatus(`Rejected: ${validation.reason}`);
        await refresh();
        return;
      }

      const signer = await provider.getSigner();
      const c = await readContract(signer);
      const safetyBuffer = ethers.parseEther("0.001");
      const payment = exactDebt + safetyBuffer;

      setStatus(`Repaying Loan #${loanId} with a safety buffer...`);
      const tx = await c.repayFull(loanId, { value: payment });
      setStatus(`Transaction submitted (${tx.hash}). Waiting for confirmation...`);
      await tx.wait();
      // Give the local node one tick to expose the mined state before reading it.
      await new Promise(resolve => setTimeout(resolve, 250));

      if (validationId) {
        await fetch(`${API_BASE}/blockchain/transactions/${validationId}/confirm`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ txHash: tx.hash })
        });
      }

      setRepayInput(x => ({ ...x, [loanId]: "" }));
      await refresh();

      const verifyProvider = new ethers.BrowserProvider(window.ethereum);
      const verifyContract = await readContract(verifyProvider);
      const remainingDebt = await verifyContract.getLoanDebt(loanId);

      if (remainingDebt === 0n) {
        setStatus(`Loan #${loanId} fully repaid and collateral released.`);
      } else {
        setStatus(`Loan #${loanId} transaction confirmed, remaining debt is ${ethers.formatEther(remainingDebt)} ETH.`);
      }
    } catch (err) {
      console.error(err);
      const message = err?.shortMessage || err?.reason || err?.info?.error?.message || err?.message || "Transaction failed";
      setStatus(`Full repayment failed: ${message}`);
      if (validationId) {
        try {
          await fetch(`${API_BASE}/blockchain/transactions/${validationId}/fail`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reason: message })
          });
        } catch (_) {}
      }
      await refresh();
    }
  };

  const executeAction = async (action, amount = "0", collateral = "0", loanId = null) => {
    if (!account) return setStatus("Please connect your wallet first.");
    let validationId = null;
    try {
      setStatus(`Checking ${ACTION_LABELS[action] || action}...`);
      const validationRes = await fetch(`${API_BASE}/blockchain/process-transaction`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ wallet: account, amount, collateral, action })
      });
      const validation = await validationRes.json();
      setIntermediateData(validation);
      validationId = validation.transactionId;

      if (!validationRes.ok || !validation.success) {
        setStatus(`Rejected: ${validation.reason}`);
        await refresh();
        return;
      }

      const p = new ethers.BrowserProvider(window.ethereum);
      const signer = await p.getSigner();
      const c = await readContract(signer);
      let tx;
      if (action === "SupplyLiquidity") tx = await c.supplyLiquidity({ value: ethers.parseEther(String(amount || "0")) });
      else if (action === "WithdrawSupply") tx = await c.withdrawSupply(ethers.parseEther(String(amount || "0")));
      else if (action === "DepositCollateral") tx = await c.depositCollateral({ value: ethers.parseEther(String(collateral || "0")) });
      else if (action === "WithdrawCollateral") tx = await c.withdrawAvailableCollateral(ethers.parseEther(String(amount || "0")));
      else if (action === "Borrow") tx = await c.borrow(ethers.parseEther(String(amount || "0")));
      else if (action === "Repay") tx = await c.repay(loanId, { value: ethers.parseEther(String(amount || "0")) });
      else if (action === "ClaimInterest") tx = await c.claimInterest();
      else throw new Error("Unsupported action");

      setStatus(`Transaction submitted (${tx.hash}). Waiting for confirmation...`);
      await tx.wait();
      // Give the local node one tick to expose the mined state before reading it.
      await new Promise(resolve => setTimeout(resolve, 250));

      if (validationId) {
        await fetch(`${API_BASE}/blockchain/transactions/${validationId}/confirm`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ txHash: tx.hash })
        });
      }

      setStatus(`${ACTION_LABELS[action] || action} confirmed on-chain.`);
      setSupplyInput(""); setWithdrawInput(""); setCollateralInput(""); setCollateralWithdrawInput(""); setBorrowInput("");
      await refresh();
    } catch (err) {
      console.error(err);
      const message = err?.shortMessage || err?.reason || err?.info?.error?.message || err?.message || "Transaction failed";
      setStatus(`Transaction failed: ${message}`);
      if (validationId) {
        try {
          await fetch(`${API_BASE}/blockchain/transactions/${validationId}/fail`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reason: message })
          });
        } catch (_) {}
      }
      await refresh();
    }
  };

  const estimatedPendingInterest = lender.sharePercent > 0 ? (platform.accruedInterest * lender.sharePercent) / 100 : 0;
  const totalLenderEarnings = lender.interest + estimatedPendingInterest;
  const risk = (ratio) => ratio === 0 ? { text: "NO DEBT", tone: "#10b981" } : ratio >= 200 ? { text: "LOW", tone: "#10b981" } : ratio >= 170 ? { text: "MEDIUM", tone: "#f59e0b" } : ratio >= 150 ? { text: "HIGH", tone: "#fb923c" } : { text: "LIQUIDATION RISK", tone: "#ef4444" };

  return (
    <div style={{ background: "#0b0f19", color: "#f8fafc", minHeight: "100vh", padding: 24, fontFamily: "Arial, sans-serif" }}>
      {/* Top Header */}
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid #1e293b", paddingBottom: 16 }}>
        <div>
          <h2 style={{ margin: 0, color: "#38bdf8" }}>DeFiLens: Integrated Decision Support Platform</h2>
          <small style={{ color: "#94a3b8" }}>Unified Blockchain Protocol, XGBoost Fraud Detection, Risk Models, & DocAI</small>
        </div>
        <Button onClick={connectWallet}>{account ? short(account) : "Connect Wallet"}</Button>
      </header>

      {/* 4 Core Architecture Tabs */}
      <div style={{ display: "flex", gap: 10, margin: "20px 0" }}>
        {[
          { id: "blockchain", label: "BLOCKCHAIN PROTOCOL" },
          { id: "fraud", label: "FRAUD DETECTION (XGBoost)" },
          { id: "risk", label: "RISK SCORING (Random Forest)" },
          { id: "docai", label: "DOCUMENT INTELLIGENCE (OCR)" }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveModule(tab.id)}
            style={{
              flex: 1,
              padding: "12px",
              borderRadius: 8,
              border: "1px solid #334155",
              background: activeModule === tab.id ? "#2563eb" : "#151d2e",
              color: "#fff",
              cursor: "pointer",
              fontWeight: 700,
              transition: "0.2s background"
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ==================== MODULE 1: BLOCKCHAIN PROTOCOL ==================== */}
      {activeModule === "blockchain" && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 20 }}>
            <Card title="Available Pool Liquidity" value={`${platform.available.toFixed(4)} ETH`} />
            <Card title="Total Supplied" value={`${platform.supplied.toFixed(4)} ETH`} accent="#10b981" />
            <Card title="Total Borrowed" value={`${platform.borrowed.toFixed(4)} ETH`} accent="#f59e0b" />
            <Card title="Utilization" value={`${platform.utilization.toFixed(0)}%`} accent="#a78bfa" />
          </div>

          <nav style={{ display: "flex", gap: 10, marginBottom: 20 }}>
            <button onClick={() => setRoute("borrower")} style={{ flex: 1, padding: 13, background: page === "borrower" ? "#2563eb" : "#1e293b", color: "#fff", border: "1px solid #334155", borderRadius: 7, fontWeight: 700, cursor: "pointer" }}>Borrower View</button>
            <button onClick={() => setRoute("lender")} style={{ flex: 1, padding: 13, background: page === "lender" ? "#10b981" : "#1e293b", color: "#fff", border: "1px solid #334155", borderRadius: 7, fontWeight: 700, cursor: "pointer" }}>Lender View</button>
          </nav>

          {page === "lender" ? (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 18 }}>
                <Card title="Your Supplied Principal" value={`${lender.principal.toFixed(4)} ETH`} accent="#10b981" />
                <Card title="Pool Share" value={`${lender.sharePercent.toFixed(2)}%`} />
                <Card title="Claimable Interest" value={`${lender.interest.toFixed(6)} ETH`} accent="#f59e0b" />
                <Card title="Estimated Pending Interest" value={`${estimatedPendingInterest.toFixed(6)} ETH`} accent="#a78bfa" sub="From currently accruing active loans" />
              </div>

              <div style={{ marginBottom: 18, padding: 12, background: "#0b0f19", borderRadius: 8, border: "1px solid #24324f", color: "#94a3b8", fontSize: 13 }}>
                <b style={{ color: "#10b981" }}>Pool accounting:</b> Available Pool Liquidity belongs to lenders only.
                Borrower collateral is kept separate. Borrower repayment returns principal and realized interest to
                the lender pool; claiming interest transfers that interest from the pool to the lender's wallet.
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: 18 }}>
                <section style={{ background: "#151d2e", padding: 20, borderRadius: 10, border: "1px solid #24324f" }}>
                  <h3>Lender Operations</h3>
                  <p style={{ color: "#94a3b8" }}>Supply ETH to the shared pool. Borrower interest is distributed to lenders according to pool shares when interest is repaid.</p>
                  <div style={{ display: "flex", gap: 10 }}>
                    <input type="number" min="0" placeholder="Supply ETH" value={supplyInput} onChange={e => setSupplyInput(e.target.value)} style={inputStyle} />
                    <Button tone="green" disabled={!supplyInput} onClick={() => executeAction("SupplyLiquidity", supplyInput)}>Supply</Button>
                  </div>
                  <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
                    <input type="number" min="0" placeholder="Withdraw principal" value={withdrawInput} onChange={e => setWithdrawInput(e.target.value)} style={inputStyle} />
                    <Button tone="gray" disabled={!withdrawInput} onClick={() => executeAction("WithdrawSupply", withdrawInput)}>Withdraw</Button>
                  </div>
                  <Button tone="orange" disabled={lender.interest <= 0} onClick={() => executeAction("ClaimInterest")} style={{ marginTop: 12 }}>Claim Interest</Button>
                  <div style={{ marginTop: 18, padding: 14, background: "#0b0f19", borderRadius: 8 }}>
                    <b>Total estimated lender earnings:</b> <span style={{ color: "#10b981" }}>{totalLenderEarnings.toFixed(6)} ETH</span>
                  </div>
                </section>

                <section style={panelStyle}>
                  <h3>How Your Yield Works</h3>
                  <p style={{ color: "#94a3b8" }}>Example: a borrower takes 10 ETH at 5% APR for 30 days.</p>
                  <div style={{ fontSize: 15, lineHeight: 1.8 }}>
                    <div>Principal: <b>10 ETH</b></div>
                    <div>Interest: <b>10 × 5% × 30 / 365 = 0.041095 ETH</b></div>
                    <div>Total borrower repayment: <b>10.041095 ETH</b></div>
                    <div style={{ color: "#10b981" }}>That realized interest becomes pool lender yield.</div>
                  </div>
                </section>
              </div>
            </>
          ) : (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12, marginBottom: 18 }}>
                <Card title="Available Collateral" value={`${borrower.availableCollateral.toFixed(4)} ETH`} accent="#10b981" />
                <Card title="Locked Collateral" value={`${borrower.lockedCollateral.toFixed(4)} ETH`} accent="#38bdf8" />
                <Card title="Total Debt" value={`${borrower.debt.toFixed(6)} ETH`} accent="#ef4444" />
                <Card title="Max Additional Borrow" value={`${Math.min(borrower.maxAdditionalBorrow, platform.available).toFixed(4)} ETH`} accent="#f59e0b" />
                <Card title="Health Ratio" value={`${borrower.healthRatio.toFixed(0)}%`} accent={risk(borrower.healthRatio).tone} sub={risk(borrower.healthRatio).text} />
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1.05fr 1fr", gap: 18 }}>
                <section style={panelStyle}>
                  <h3>Borrower Operations</h3>
                  <p style={{ color: "#94a3b8" }}>Deposit collateral first. Borrowing checks real pool liquidity and your remaining collateral capacity on-chain.</p>
                  <div style={{ display: "flex", gap: 10 }}>
                    <input type="number" min="0" placeholder="Collateral ETH" value={collateralInput} onChange={e => setCollateralInput(e.target.value)} style={inputStyle} />
                    <Button tone="green" disabled={!collateralInput} onClick={() => executeAction("DepositCollateral", "0", collateralInput)}>Deposit Collateral</Button>
                  </div>
                  <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
                    <input type="number" min="0" placeholder="Withdraw unused collateral" value={collateralWithdrawInput} onChange={e => setCollateralWithdrawInput(e.target.value)} style={inputStyle} />
                    <Button tone="gray" disabled={!collateralWithdrawInput} onClick={() => executeAction("WithdrawCollateral", collateralWithdrawInput)}>Withdraw</Button>
                  </div>
                  <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
                    <input type="number" min="0" placeholder="Borrow ETH" value={borrowInput} onChange={e => setBorrowInput(e.target.value)} style={inputStyle} />
                    <Button disabled={!borrowInput} onClick={() => executeAction("Borrow", borrowInput)}>Take Loan</Button>
                  </div>
                  <div style={{ marginTop: 14, color: "#94a3b8", fontSize: 13 }}>Current fixed rate: <b style={{ color: "#f59e0b" }}>5% APR</b> · Collateral ratio: <b>150%</b> · Repay anytime</div>
                </section>

                <section style={panelStyle}>
                  <h3>Borrowing Conditions</h3>
                  <div style={{ display: "grid", gap: 10 }}>
                    {infoRow("Pool available", `${platform.available.toFixed(6)} ETH`)}
                    {infoRow("Your total collateral", `${(borrower.availableCollateral + borrower.lockedCollateral).toFixed(6)} ETH`)}
                    {infoRow("Existing debt", `${borrower.debt.toFixed(6)} ETH`)}
                    {infoRow("Maximum additional", `${Math.min(borrower.maxAdditionalBorrow, platform.available).toFixed(6)} ETH`)}
                  </div>
                  <p style={{ color: "#94a3b8", marginTop: 14 }}>Collateral for a new loan is locked at 150% of that loan's principal. Full repayment moves locked collateral back to Available Collateral; withdraw it separately when needed.</p>
                </section>
              </div>

              <section style={{ ...panelStyle, marginTop: 18 }}>
                <h3>My Active Loans</h3>
                {loans.filter(l => l.active).length === 0 ? <p style={{ color: "#64748b" }}>No active loans.</p> : (
                  <div style={{ overflowX: "auto" }}>
                    <table style={tableStyle}>
                      <thead><tr><th>Loan</th><th>Borrowed</th><th>Collateral</th><th>Interest</th><th>Rate</th><th>Health</th><th>Started</th><th>Repay</th></tr></thead>
                      <tbody>
                        {loans.filter(l => l.active).map(loan => {
                          const debt = loan.principal + loan.interest;
                          const health = debt > 0 ? (loan.collateral / debt) * 100 : 0;
                          const r = risk(health);
                          return (
                            <tr key={loan.id}>
                              <td>#{loan.id}</td><td>{loan.principal.toFixed(6)} ETH</td><td>{loan.collateral.toFixed(6)} ETH</td>
                              <td>{loan.interest.toFixed(8)} ETH</td><td>{loan.rate}% APR</td>
                              <td><span style={{ color: r.tone, fontWeight: 700 }}>{health.toFixed(1)}% · {r.text}</span></td>
                              <td>{dateText(loan.startTime)}</td>
                              <td>
                                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                                  <input
                                    type="number"
                                    min="0"
                                    placeholder="Repay amount"
                                    value={repayInput[loan.id] || ""}
                                    onChange={e => setRepayInput(x => ({ ...x, [loan.id]: e.target.value }))}
                                    style={{ ...inputStyle, width: 130 }}
                                  />
                                  <Button tone="orange" disabled={!repayInput[loan.id]} onClick={() => executeAction("Repay", repayInput[loan.id], "0", loan.id)}>Repay</Button>
                                  <Button tone="green" onClick={() => repayFull(loan.id)}>Repay Full</Button>
                                </div>
                                <small style={{ color: "#64748b" }}>Full repayment releases {loan.collateral.toFixed(6)} ETH</small>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>
            </>
          )}

          <section style={{ ...panelStyle, marginTop: 18 }}>
            <h3>Transaction Status & Pre-Validation State</h3>
            <p style={{ color: status.toLowerCase().includes("failed") || status.toLowerCase().includes("rejected") ? "#ef4444" : "#10b981" }}>{status}</p>
            {lastUpdated && <small style={{ color: "#64748b" }}>Blockchain data refreshed: {lastUpdated.toLocaleTimeString()}</small>}
            {intermediateData && <pre style={{ background: "#050811", padding: 12, borderRadius: 7, color: "#38bdf8", overflowX: "auto" }}>{JSON.stringify(intermediateData, null, 2)}</pre>}
          </section>

          <section style={{ ...panelStyle, marginTop: 18 }}>
            <h3>Platform Audit Trail (MySQL Ledger)</h3>
            <div style={{ overflowX: "auto" }}>
              <table style={tableStyle}>
                <thead><tr><th>ID</th><th>Wallet</th><th>Action</th><th>Amount</th><th>Collateral</th><th>Status</th><th>Tx Hash</th><th>Notes</th></tr></thead>
                <tbody>{dbTransactions.map(tx => (
                  <tr key={tx.id}><td>#{tx.id}</td><td>{short(tx.wallet_address)}</td><td>{tx.action_type}</td><td>{tx.borrow_amount} ETH</td><td>{tx.collateral_amount} ETH</td><td style={{ color: tx.status === "CONFIRMED" ? "#10b981" : tx.status === "FAILED" || tx.status === "REJECTED" ? "#ef4444" : "#f59e0b" }}>{tx.status}</td><td>{tx.tx_hash ? short(tx.tx_hash) : "-"}</td><td>{tx.rejection_reason || "-"}</td></tr>
                ))}</tbody>
              </table>
            </div>
          </section>
        </>
      )}

      {/* ==================== MODULE 2: FRAUD DETECTION ==================== */}
      {activeModule === "fraud" && (
        <section style={{ background: "#151d2e", borderRadius: 10, overflow: "hidden", border: "1px solid #24324f" }}>
          <div style={{ padding: "14px 20px", background: "#101726", borderBottom: "1px solid #24324f", display: "flex", justifyContent: "space-between" }}>
            <span style={{ fontWeight: 700, color: "#38bdf8" }}>XGBoost Classifier & SHAP Explainability Engine</span>
            <small style={{ color: "#64748b" }}>Live on http://localhost:8501</small>
          </div>
          <iframe
            src="http://localhost:8501/?embed=true"
            title="Fraud Module"
            style={{ width: "100%", height: "850px", border: "none", background: "#0e1117" }}
          />
        </section>
      )}

      {/* ==================== MODULE 3: RISK SCORING ==================== */}
      {activeModule === "risk" && (
        <section style={{ background: "#151d2e", borderRadius: 10, overflow: "hidden", border: "1px solid #24324f" }}>
          <div style={{ padding: "14px 20px", background: "#101726", borderBottom: "1px solid #24324f", display: "flex", justifyContent: "space-between" }}>
            <span style={{ fontWeight: 700, color: "#10b981" }}>Random Forest Default Risk & Dynamic LTV Scoring</span>
            <small style={{ color: "#64748b" }}>Live on http://localhost:8502</small>
          </div>
          <iframe
            src="http://localhost:8502/?embed=true"
            title="Risk Module"
            style={{ width: "100%", height: "850px", border: "none", background: "#0e1117" }}
          />
        </section>
      )}

      {/* ==================== MODULE 4: DOCUMENT INTELLIGENCE ==================== */}
      {activeModule === "docai" && (
        <section style={{ background: "#151d2e", borderRadius: 10, overflow: "hidden", border: "1px solid #24324f" }}>
          <div style={{ padding: "14px 20px", background: "#101726", borderBottom: "1px solid #24324f", display: "flex", justifyContent: "space-between" }}>
            <span style={{ fontWeight: 700, color: "#f59e0b" }}>Document OCR & Financial Statement Intelligence</span>
            <small style={{ color: "#64748b" }}>Live on http://localhost:8503</small>
          </div>
          <iframe
            src="http://localhost:8503/?embed=true"
            title="DocAI Module"
            style={{ width: "100%", height: "850px", border: "none", background: "#0e1117" }}
          />
        </section>
      )}
    </div>
  );
}

const inputStyle = { flex: 1, padding: "10px", background: "#0b0f19", border: "1px solid #334155", color: "#fff", borderRadius: 6 };
const panelStyle = { background: "#151d2e", padding: 20, borderRadius: 10, border: "1px solid #24324f" };
const tableStyle = { width: "100%", textAlign: "left", borderCollapse: "collapse", fontSize: 13 };
const infoRow = (a, b) => (
  <div style={{ display: "flex", justifyContent: "space-between", padding: 10, background: "#0b0f19", borderRadius: 6 }}>
    <span style={{ color: "#94a3b8" }}>{a}</span>
    <b>{b}</b>
  </div>
);