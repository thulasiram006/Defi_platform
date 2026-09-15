import { network } from "hardhat";

const CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3";

async function main() {
  const { ethers } = await network.connect();
  const lending = await ethers.getContractAt("Lending", CONTRACT_ADDRESS);

  console.log("\n================================");
  console.log("⛓️ FETCHING ALL ON-CHAIN LOANS");
  console.log("================ lash\n");

  const allLoans = await lending.getAllLoans();
  console.log("Total Loans Count:", allLoans.length);

  allLoans.forEach((loan, index) => {
    console.log(`\n--- Loan #${index} ---`);
    console.log("Borrower:", loan.borrower);
    console.log("Principal:", ethers.formatEther(loan.principal), "ETH");
    console.log("Collateral:", ethers.formatEther(loan.collateral), "ETH");
    console.log("Active:", loan.active);
  });
  console.log("\n================ lash\n");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
