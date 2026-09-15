// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Lending {
    uint256 public constant COLLATERAL_RATIO_PERCENT = 150;
    uint256 public constant ANNUAL_INTEREST_RATE = 5;
    uint256 private constant INDEX_SCALE = 1e18;

    error InvalidAmount();
    error InsufficientCollateral();
    error InsufficientPoolLiquidity();
    error NoActiveDebt();
    error ExcessiveRepayment();
    error InsufficientRepayment();
    error InsufficientLenderBalance();
    error InsufficientAvailableCollateral();
    error LoanNotFound();

    struct Loan {
        uint256 id;
        address borrower;
        uint256 principal;
        uint256 collateral;
        uint256 interestRate;
        uint256 startTime;
        uint256 lastAccrualTime;
        uint256 accruedInterest;
        bool active;
    }

    // =========================
    // LENDER ACCOUNTING
    // =========================

    mapping(address => uint256) public lenderPrincipal;
    mapping(address => uint256) public lenderShares;
    mapping(address => uint256) public lenderInterest;
    mapping(address => uint256) public lenderIndex;

    uint256 public totalLenderPrincipal;
    uint256 public totalLenderShares;
    uint256 public lenderInterestIndex;

    // Total realized lender interest that is still inside the pool.
    // It increases when borrowers repay interest and decreases when
    // lenders claim that interest.
    uint256 public totalClaimableInterest;

    // =========================
    // BORROWER COLLATERAL
    // =========================

    mapping(address => uint256) public availableCollateral;

    // Total borrower collateral held by the contract, both available
    // and locked. This amount is NEVER part of lender pool liquidity.
    uint256 public totalCollateralDeposited;

    // =========================
    // LOAN ACCOUNTING
    // =========================

    uint256 public totalBorrowed;
    uint256 public totalCollateralLocked;
    uint256 public nextLoanId = 1;

    mapping(uint256 => Loan) private loans;
    mapping(address => uint256[]) private borrowerLoanIds;
    uint256[] private allLoanIds;

    // =========================
    // EVENTS
    // =========================

    event LiquiditySupplied(
        address indexed lender,
        uint256 amount,
        uint256 shares
    );

    event LiquidityWithdrawn(
        address indexed lender,
        uint256 amount
    );

    event InterestClaimed(
        address indexed lender,
        uint256 amount
    );

    event CollateralDeposited(
        address indexed borrower,
        uint256 amount
    );

    event CollateralWithdrawn(
        address indexed borrower,
        uint256 amount
    );

    event LoanCreated(
        address indexed borrower,
        uint256 indexed loanId,
        uint256 principal,
        uint256 collateral,
        uint256 interestRate
    );

    event LoanRepaid(
        address indexed borrower,
        uint256 indexed loanId,
        uint256 payment,
        uint256 interestPaid,
        uint256 principalPaid,
        uint256 collateralReleased
    );

    // =========================
    // LENDER FUNCTIONS
    // =========================

    function supplyLiquidity() external payable {
        if (msg.value == 0) revert InvalidAmount();

        _settleLender(msg.sender);

        uint256 shares;

        if (totalLenderShares == 0) {
            shares = msg.value;
        } else {
            uint256 assets = poolAssets();

            if (assets == 0) revert InvalidAmount();

            shares = (msg.value * totalLenderShares) / assets;

            if (shares == 0) {
                shares = 1;
            }
        }

        lenderPrincipal[msg.sender] += msg.value;
        lenderShares[msg.sender] += shares;

        totalLenderPrincipal += msg.value;
        totalLenderShares += shares;

        lenderIndex[msg.sender] = lenderInterestIndex;

        emit LiquiditySupplied(
            msg.sender,
            msg.value,
            shares
        );
    }

    function withdrawSupply(uint256 amount) external {
        if (amount == 0) revert InvalidAmount();

        _settleLender(msg.sender);

        if (lenderPrincipal[msg.sender] < amount) {
            revert InsufficientLenderBalance();
        }

        if (availableLiquidity() < amount) {
            revert InsufficientPoolLiquidity();
        }

        uint256 principalBefore = lenderPrincipal[msg.sender];

        uint256 sharesToBurn =
            (lenderShares[msg.sender] * amount) /
            principalBefore;

        if (sharesToBurn == 0) {
            sharesToBurn = 1;
        }

        if (sharesToBurn > lenderShares[msg.sender]) {
            sharesToBurn = lenderShares[msg.sender];
        }

        lenderPrincipal[msg.sender] -= amount;
        lenderShares[msg.sender] -= sharesToBurn;

        totalLenderPrincipal -= amount;
        totalLenderShares -= sharesToBurn;

        lenderIndex[msg.sender] = lenderInterestIndex;

        (bool sent, ) =
            payable(msg.sender).call{value: amount}("");

        require(sent, "ETH transfer failed");

        emit LiquidityWithdrawn(
            msg.sender,
            amount
        );
    }

    function claimInterest() external {
        _settleLender(msg.sender);

        uint256 amount = lenderInterest[msg.sender];

        if (amount == 0) {
            revert InvalidAmount();
        }

        if (availableLiquidity() < amount) {
            revert InsufficientPoolLiquidity();
        }

        lenderInterest[msg.sender] = 0;
        totalClaimableInterest -= amount;

        (bool sent, ) =
            payable(msg.sender).call{value: amount}("");

        require(sent, "Interest transfer failed");

        emit InterestClaimed(
            msg.sender,
            amount
        );
    }

    // =========================
    // COLLATERAL
    // =========================

    function depositCollateral() external payable {
        if (msg.value == 0) {
            revert InvalidAmount();
        }

        availableCollateral[msg.sender] += msg.value;
        totalCollateralDeposited += msg.value;

        emit CollateralDeposited(
            msg.sender,
            msg.value
        );
    }

    function withdrawAvailableCollateral(
        uint256 amount
    ) external {
        if (amount == 0) {
            revert InvalidAmount();
        }

        if (availableCollateral[msg.sender] < amount) {
            revert InsufficientAvailableCollateral();
        }

        availableCollateral[msg.sender] -= amount;
        totalCollateralDeposited -= amount;

        (bool sent, ) =
            payable(msg.sender).call{value: amount}("");

        require(sent, "Collateral withdrawal failed");

        emit CollateralWithdrawn(msg.sender, amount);
    }

    // =========================
    // BORROW
    // =========================

    function borrow(uint256 amount) external {
        if (amount == 0) {
            revert InvalidAmount();
        }

        if (availableLiquidity() < amount) {
            revert InsufficientPoolLiquidity();
        }

        uint256 requiredCollateral =
            (amount * COLLATERAL_RATIO_PERCENT) / 100;

        if (availableCollateral[msg.sender] < requiredCollateral) {
            revert InsufficientCollateral();
        }

        availableCollateral[msg.sender] -= requiredCollateral;
        totalCollateralLocked += requiredCollateral;

        uint256 loanId = nextLoanId++;

        loans[loanId] = Loan({
            id: loanId,
            borrower: msg.sender,
            principal: amount,
            collateral: requiredCollateral,
            interestRate: ANNUAL_INTEREST_RATE,
            startTime: block.timestamp,
            lastAccrualTime: block.timestamp,
            accruedInterest: 0,
            active: true
        });

        borrowerLoanIds[msg.sender].push(loanId);
        allLoanIds.push(loanId);

        totalBorrowed += amount;

        (bool sent, ) =
            payable(msg.sender).call{value: amount}("");

        require(sent, "Loan transfer failed");

        emit LoanCreated(
            msg.sender,
            loanId,
            amount,
            requiredCollateral,
            ANNUAL_INTEREST_RATE
        );
    }

    // =========================
    // REPAY
    // =========================

    function repay(uint256 loanId) external payable {
        if (msg.value == 0) {
            revert InvalidAmount();
        }

        Loan storage loan = loans[loanId];

        if (
            loan.id == 0 ||
            loan.borrower != msg.sender
        ) {
            revert LoanNotFound();
        }

        if (!loan.active) {
            revert NoActiveDebt();
        }

        _accrueLoan(loan);

        uint256 debt =
            loan.principal +
            loan.accruedInterest;

        if (debt == 0) {
            revert NoActiveDebt();
        }

        if (msg.value > debt) {
            revert ExcessiveRepayment();
        }

        uint256 interestPaid =
            msg.value > loan.accruedInterest
                ? loan.accruedInterest
                : msg.value;

        uint256 principalPaid =
            msg.value - interestPaid;

        loan.accruedInterest -= interestPaid;
        loan.principal -= principalPaid;

        totalBorrowed -= principalPaid;

        if (
            interestPaid > 0 &&
            totalLenderShares > 0
        ) {
            lenderInterestIndex +=
                (interestPaid * INDEX_SCALE) /
                totalLenderShares;

            totalClaimableInterest += interestPaid;
        }

        uint256 released = 0;

        // CLOSE LOAN WHEN EVERYTHING IS REPAID
        if (
            loan.principal == 0 &&
            loan.accruedInterest == 0
        ) {
            released = loan.collateral;

            loan.collateral = 0;
            loan.active = false;

            totalCollateralLocked -= released;

            // The collateral is released from the loan and becomes
            // available to the borrower. It stays inside the contract
            // until the borrower explicitly withdraws it.
            availableCollateral[msg.sender] += released;
        }

        emit LoanRepaid(
            msg.sender,
            loanId,
            msg.value,
            interestPaid,
            principalPaid,
            released
        );
    }

    // =========================
    // EXACT LOAN DEBT
    // =========================

    function getLoanDebt(
        uint256 loanId
    ) external view returns (uint256) {
        Loan storage loan = loans[loanId];

        if (loan.id == 0) {
            revert LoanNotFound();
        }

        if (!loan.active) {
            return 0;
        }

        uint256 interest =
            loan.accruedInterest;

        if (loan.principal > 0) {
            uint256 elapsed =
                block.timestamp -
                loan.lastAccrualTime;

            interest +=
                (
                    loan.principal *
                    loan.interestRate *
                    elapsed
                ) /
                (365 days * 100);
        }

        return loan.principal + interest;
    }
    function repayFull(uint256 loanId) external payable {
    if (msg.value == 0) revert InvalidAmount();

    Loan storage loan = loans[loanId];

    if (loan.id == 0 || loan.borrower != msg.sender) {
        revert LoanNotFound();
    }

    if (!loan.active) {
        revert NoActiveDebt();
    }

    // Calculate the debt at the exact transaction timestamp.
    _accrueLoan(loan);

    uint256 debt = loan.principal + loan.accruedInterest;

    if (debt == 0) {
        revert NoActiveDebt();
    }

    // Full repayment must cover the entire current debt.
    // The frontend can send a small safety buffer.
    if (msg.value < debt) {
        revert InsufficientRepayment();
    }

    uint256 interestPaid = loan.accruedInterest;
    uint256 principalPaid = loan.principal;

    // The complete debt is being paid.
    loan.accruedInterest = 0;
    loan.principal = 0;

    totalBorrowed -= principalPaid;

    // Distribute realized interest to lenders.
    if (interestPaid > 0 && totalLenderShares > 0) {
        lenderInterestIndex +=
            (interestPaid * INDEX_SCALE) / totalLenderShares;

        totalClaimableInterest += interestPaid;
    }

    // Release this loan's specific collateral.
    uint256 released = loan.collateral;

    loan.collateral = 0;
    loan.active = false;

    totalCollateralLocked -= released;

    // The collateral is now available to the borrower again.
    // It remains inside the contract until explicitly withdrawn.
    availableCollateral[msg.sender] += released;

    // Refund any safety-buffer ETH that was not needed.
    uint256 refund = msg.value - debt;

    if (refund > 0) {
        (bool refunded, ) = payable(msg.sender).call{value: refund}("");
        require(refunded, "Refund failed");
    }

    emit LoanRepaid(
        msg.sender,
        loanId,
        debt,
        interestPaid,
        principalPaid,
        released
    );
}
    // =========================
    // ACCRUAL
    // =========================

    function _accrueLoan(
        Loan storage loan
    ) internal {
        uint256 elapsed =
            block.timestamp -
            loan.lastAccrualTime;

        if (
            elapsed > 0 &&
            loan.principal > 0
        ) {
            uint256 interest =
                (
                    loan.principal *
                    loan.interestRate *
                    elapsed
                ) /
                (365 days * 100);

            loan.accruedInterest += interest;
        }

        loan.lastAccrualTime =
            block.timestamp;
    }

    // =========================
    // LENDER INTEREST
    // =========================

    function _settleLender(
        address lender
    ) internal {
        uint256 shares =
            lenderShares[lender];

        if (shares == 0) {
            lenderIndex[lender] =
                lenderInterestIndex;

            return;
        }

        uint256 delta =
            lenderInterestIndex -
            lenderIndex[lender];

        if (delta > 0) {
            lenderInterest[lender] +=
                (shares * delta) /
                INDEX_SCALE;
        }

        lenderIndex[lender] =
            lenderInterestIndex;
    }

    // =========================
    // POOL INFORMATION
    // =========================

    function availableLiquidity()
        public
        view
        returns (uint256)
    {
        // ONLY lender-owned pool assets are counted here.
        //
        // totalLenderPrincipal
        //   = lender principal currently supplied
        //
        // totalBorrowed
        //   = principal currently lent to borrowers
        //
        // totalClaimableInterest
        //   = realized borrower interest still held by the pool
        //
        // Borrower collateral is deliberately excluded. It is tracked
        // separately by totalCollateralDeposited.
        uint256 lenderAssets =
            totalLenderPrincipal +
            totalClaimableInterest;

        if (lenderAssets <= totalBorrowed) {
            return 0;
        }

        return lenderAssets - totalBorrowed;
    }

    // Explicit read-only getter used by the frontend.
    // It returns the latest lender-owned pool liquidity.
    function getAvailablePoolLiquidity()
        external
        view
        returns (uint256)
    {
        return availableLiquidity();
    }

    function totalAccruedInterest()
        public
        view
        returns (uint256 total)
    {
        for (
            uint256 i = 0;
            i < allLoanIds.length;
            i++
        ) {
            Loan storage loan =
                loans[allLoanIds[i]];

            if (!loan.active) {
                continue;
            }

            if (loan.principal == 0) {
                total += loan.accruedInterest;
                continue;
            }

            uint256 current =
                loan.accruedInterest;

            uint256 elapsed =
                block.timestamp -
                loan.lastAccrualTime;

            current +=
                (
                    loan.principal *
                    loan.interestRate *
                    elapsed
                ) /
                (365 days * 100);

            total += current;
        }
    }

    function poolAssets()
        public
        view
        returns (uint256)
    {
        return
            availableLiquidity() +
            totalBorrowed +
            totalAccruedInterest();
    }

    // =========================
    // LOAN INFORMATION
    // =========================

    function getLoan(
        uint256 loanId
    )
        external
        view
        returns (Loan memory loan)
    {
        loan = loans[loanId];

        if (loan.id == 0) {
            revert LoanNotFound();
        }

        if (loan.active) {
            uint256 elapsed =
                block.timestamp -
                loan.lastAccrualTime;

            if (loan.principal > 0) {
                loan.accruedInterest +=
                    (
                        loan.principal *
                        loan.interestRate *
                        elapsed
                    ) /
                    (365 days * 100);
            }
        }
    }

    function getBorrowerLoans(
        address borrower
    )
        external
        view
        returns (Loan[] memory result)
    {
        uint256[] storage ids =
            borrowerLoanIds[borrower];

        result = new Loan[](ids.length);

        for (
            uint256 i = 0;
            i < ids.length;
            i++
        ) {
            Loan memory loan =
                loans[ids[i]];

            if (loan.active) {
                uint256 elapsed =
                    block.timestamp -
                    loan.lastAccrualTime;

                if (loan.principal > 0) {
                    loan.accruedInterest +=
                        (
                            loan.principal *
                            loan.interestRate *
                            elapsed
                        ) /
                        (365 days * 100);
                }
            }

            result[i] = loan;
        }
    }

    function getAllLoans()
        external
        view
        returns (Loan[] memory result)
    {
        result =
            new Loan[](allLoanIds.length);

        for (
            uint256 i = 0;
            i < allLoanIds.length;
            i++
        ) {
            Loan memory loan =
                loans[allLoanIds[i]];

            if (loan.active) {
                uint256 elapsed =
                    block.timestamp -
                    loan.lastAccrualTime;

                if (loan.principal > 0) {
                    loan.accruedInterest +=
                        (
                            loan.principal *
                            loan.interestRate *
                            elapsed
                        ) /
                        (365 days * 100);
                }
            }

            result[i] = loan;
        }
    }

    // =========================
    // LENDER INFORMATION
    // =========================

    function getLenderInfo(
        address lender
    )
        external
        view
        returns (
            uint256 principal,
            uint256 claimableInterest,
            uint256 shares,
            uint256 poolSharePercent
        )
    {
        principal =
            lenderPrincipal[lender];

        shares =
            lenderShares[lender];

        claimableInterest =
            lenderInterest[lender];

        if (shares > 0) {
            claimableInterest +=
                (
                    shares *
                    (
                        lenderInterestIndex -
                        lenderIndex[lender]
                    )
                ) /
                INDEX_SCALE;
        }

        if (totalLenderShares > 0) {
            poolSharePercent =
                (shares * 10000) /
                totalLenderShares;
        }
    }

    // =========================
    // BORROWER SUMMARY
    // =========================

    function getBorrowerSummary(
        address borrower
    )
        external
        view
        returns (
            uint256 availableCol,
            uint256 lockedCol,
            uint256 debt,
            uint256 maxAdditionalBorrow,
            uint256 healthRatioPercent
        )
    {
        availableCol =
            availableCollateral[borrower];

        uint256[] storage ids =
            borrowerLoanIds[borrower];

        for (
            uint256 i = 0;
            i < ids.length;
            i++
        ) {
            Loan memory loan =
                loans[ids[i]];

            // IMPORTANT:
            // Only inactive loans are ignored.
            // Even a zero-principal active loan
            // still has locked collateral.

            if (!loan.active) {
                continue;
            }

            uint256 interest =
                loan.accruedInterest;

            if (loan.principal > 0) {
                uint256 elapsed =
                    block.timestamp -
                    loan.lastAccrualTime;

                interest +=
                    (
                        loan.principal *
                        loan.interestRate *
                        elapsed
                    ) /
                    (365 days * 100);
            }

            lockedCol +=
                loan.collateral;

            debt +=
                loan.principal +
                interest;
        }

        uint256 totalCol =
            availableCol +
            lockedCol;

        uint256 maxTotalDebt =
            (totalCol * 100) /
            COLLATERAL_RATIO_PERCENT;

        maxAdditionalBorrow =
            maxTotalDebt > debt
                ? maxTotalDebt - debt
                : 0;

        // Avoid absurd health ratios when debt is zero.
        healthRatioPercent =
            debt > 0
                ? (lockedCol * 100) / debt
                : 0;
    }

    // =========================
    // PLATFORM STATISTICS
    // =========================

    function getPlatformStats()
        external
        view
        returns (
            uint256 available,
            uint256 supplied,
            uint256 borrowed,
            uint256 collateral,
            uint256 accruedInterest,
            uint256 utilizationPercent
        )
    {
        available =
            availableLiquidity();

        supplied =
            totalLenderPrincipal;

        borrowed =
            totalBorrowed;

        collateral =
            totalCollateralLocked;

        accruedInterest =
            totalAccruedInterest();

        uint256 totalCredit =
            available +
            borrowed;

        utilizationPercent =
            totalCredit > 0
                ? (borrowed * 100) /
                    totalCredit
                : 0;
    }
}