import { network } from "hardhat";

async function main() {
  const { ethers } = await network.connect();
  const contractAddress = "0xe7f1725e7734ce288f8367e1bb143e90bb3f0512";

  const lending = await ethers.getContractAt("Lending", contractAddress);

  console.log("1. Supplying pool liquidity (10 ETH)...");
  const sTx = await lending.supplyLiquidity({ value: ethers.parseEther("10.0") });
  await sTx.wait();

  console.log("2. Depositing collateral (5 ETH)...");
  const dTx = await lending.depositCollateral({ value: ethers.parseEther("5.0") });
  await dTx.wait();
    
  console.log("3. Executing borrow (1.5 ETH)...");
  const bTx = await lending.borrow(ethers.parseEther("1.5"));
  await bTx.wait();

  console.log("✅ Borrow successful! Tx Hash:", bTx.hash);
}

main().catch((err) => console.error(err));