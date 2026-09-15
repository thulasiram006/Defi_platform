import { network } from "hardhat";

async function main() {
  const { ethers } = await network.connect();

  const contractAddress = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512";
  const [signer] = await ethers.getSigners();
  const borrower = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e";

  const lending = await ethers.getContractAt(
    "Lending",
    contractAddress
  );

  console.log("⏳ Testing createLoan call directly on-chain...");

  try {
    const tx = await lending.createLoan(
      borrower,
      ethers.parseEther("1.0"),
      ethers.parseEther("2.0")
    );
    await tx.wait();
    console.log("✅ Success! Tx Hash:", tx.hash);
  } catch (error) {
    console.error("❌ Exact Revert Details:", error);
  }
}

main().catch((err) => console.error(err));