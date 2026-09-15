import { network } from "hardhat";

async function main() {
  const { ethers } = await network.connect();
  const contractAddress = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512";

  const lending = await ethers.getContractAt("Lending", contractAddress);

  console.log("\n==================================================");
  console.log("📜 AVAILABLE CONTRACT FUNCTIONS ON LENDING.SOL:");
  console.log("==================================================\n");

  const fragmentNames = lending.interface.fragments
    .filter((f) => f.type === "function")
    .map((f) => f.name);

  console.log(fragmentNames);
  console.log("\n==================================================\n");
}

main().catch((err) => console.error(err));