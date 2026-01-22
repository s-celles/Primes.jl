# Quadratic Sieve factorization algorithm
# Reference: https://en.wikipedia.org/wiki/Quadratic_sieve
# This implements the basic quadratic sieve for factoring large integers
# with balanced prime factors (40-70 digits).

#=
Threshold for switching from Pollard's rho to Quadratic Sieve.
Numbers larger than this with balanced factors benefit from QS.
Approximately 10^30 (~100 bits).
=#
const QUADRATIC_SIEVE_THRESHOLD = big(10)^30

#=============================================================================
  Data Structures
=============================================================================#

"""
    QSFactorBase

Stores precomputed data about the factor base primes for the quadratic sieve.

# Fields
- `primes::Vector{Int}`: Factor base primes p where Legendre symbol (n|p) = 1
- `sqrt_n_mod_p::Vector{Int}`: Precomputed √n mod p for each prime
- `log_p::Vector{Float64}`: log(p) for log-sieving optimization
"""
struct QSFactorBase
    primes::Vector{Int}
    sqrt_n_mod_p::Vector{Int}
    log_p::Vector{Float64}
end

"""
    QSSmoothNumber{T<:Integer}

Represents a B-smooth number found during sieving.

# Fields
- `x::T`: The x value where Q(x) = x² - n is smooth
- `q::T`: Q(x) = x² - n (the smooth value, may be negative)
- `exponents::Vector{Int}`: Full exponent vector for each factor base prime
"""
struct QSSmoothNumber{T<:Integer}
    x::T
    q::T
    exponents::Vector{Int}
end

"""
    QSMatrix

The exponent matrix for linear algebra over GF(2).

# Fields
- `rows::Vector{BitVector}`: Each row is a smooth number's exponent parity vector
- `smooth_indices::Vector{Int}`: Maps row index to smooth number index
- `num_primes::Int`: Number of columns (factor base size)
"""
struct QSMatrix
    rows::Vector{BitVector}
    smooth_indices::Vector{Int}
    num_primes::Int
end

#=============================================================================
  Parameter Selection
=============================================================================#

"""
    compute_factor_base_bound(n::Integer) -> Int

Compute the optimal factor base bound B using L-notation formula.
B ≈ L(n)^(1/√2) where L(n) = exp(√(ln(n) * ln(ln(n))))

For a 60-digit number, this gives B ≈ 460,000.
"""
function compute_factor_base_bound(n::Integer)
    ln_n = log(Float64(n))
    ln_ln_n = log(ln_n)
    L_n = exp(sqrt(ln_n * ln_ln_n))
    B = L_n^(1/sqrt(2))
    return max(100, min(Int(ceil(B)), 10_000_000))  # Clamp to reasonable range
end

"""
    compute_sieve_interval(n::Integer) -> Int

Compute the sieving interval half-width M using L-notation formula.
M ≈ L(n)^√2

For a 60-digit number, this gives M ≈ 10^8.
"""
function compute_sieve_interval(n::Integer)
    ln_n = log(Float64(n))
    ln_ln_n = log(ln_n)
    L_n = exp(sqrt(ln_n * ln_ln_n))
    M = L_n^sqrt(2)
    return max(10_000, min(Int(ceil(M)), 100_000_000))  # Clamp to reasonable range
end

#=============================================================================
  Factor Base Construction
=============================================================================#

"""
    tonelli_shanks(n::Integer, p::Integer) -> Integer

Compute √n mod p using the Tonelli-Shanks algorithm.
Assumes n is a quadratic residue mod p (Legendre symbol = 1).
"""
function tonelli_shanks(n::Integer, p::Integer)
    n = mod(n, p)
    n == 0 && return 0
    p == 2 && return n

    # Find Q and S such that p - 1 = Q * 2^S with Q odd
    Q = p - 1
    S = 0
    while iseven(Q)
        Q ÷= 2
        S += 1
    end

    # Find a quadratic non-residue z
    z = 2
    while powermod(z, (p - 1) ÷ 2, p) != p - 1
        z += 1
    end

    M = S
    c = powermod(z, Q, p)
    t = powermod(n, Q, p)
    R = powermod(n, (Q + 1) ÷ 2, p)

    while true
        t == 0 && return 0
        t == 1 && return R

        # Find the least i such that t^(2^i) = 1
        i = 1
        temp = mod(t * t, p)
        while temp != 1
            temp = mod(temp * temp, p)
            i += 1
        end

        # Update values
        b = powermod(c, 1 << (M - i - 1), p)
        M = i
        c = mod(b * b, p)
        t = mod(t * c, p)
        R = mod(R * b, p)
    end
end

"""
    build_factor_base(n::Integer, B::Int) -> QSFactorBase

Build the factor base consisting of primes p ≤ B where n is a quadratic residue mod p.
Also precomputes √n mod p for each prime.
"""
function build_factor_base(n::Integer, B::Int)
    primes_list = Int[]
    sqrt_list = Int[]
    log_list = Float64[]

    # Always include 2 if n is odd (which it should be)
    if isodd(n)
        push!(primes_list, 2)
        push!(sqrt_list, Int(mod(n, 2)))
        push!(log_list, log(2.0))
    end

    # Check each odd prime up to B
    for p in 3:2:B
        # Skip non-primes using simple trial division for small primes
        is_prime = true
        for d in 3:2:isqrt(p)
            if p % d == 0
                is_prime = false
                break
            end
        end
        !is_prime && continue

        # Check if n is a quadratic residue mod p using Euler's criterion
        leg = powermod(BigInt(n), (p - 1) ÷ 2, p)
        if leg == 1
            push!(primes_list, p)
            sqrt_n = tonelli_shanks(n, p)
            push!(sqrt_list, Int(sqrt_n))
            push!(log_list, log(Float64(p)))
        end
    end

    return QSFactorBase(primes_list, sqrt_list, log_list)
end

#=============================================================================
  Sieving Phase
=============================================================================#

"""
    init_sieve_array(n::Integer, sqrt_n::Integer, M::Int) -> Tuple{Vector{Float64}, Integer}

Initialize the sieve array for log-sieving.
Returns (sieve_array, start_x) where start_x is the first x value.
"""
function init_sieve_array(n::Integer, sqrt_n::Integer, M::Int)
    # Sieve array stores log approximations
    # We sieve Q(x) = x² - n for x in [sqrt_n, sqrt_n + M] (only positive side for simplicity)
    sieve = zeros(Float64, M + 1)

    # Initialize with log(|Q(x)|) approximations
    # Use approximation: log(Q(x)) ≈ log((sqrt_n + offset)² - n) ≈ log(2 * sqrt_n * offset)
    log_sqrt_n = log(Float64(sqrt_n))
    for i in 1:(M + 1)
        offset = i - 1
        # Q(sqrt_n + offset) = (sqrt_n + offset)² - n ≈ 2 * sqrt_n * offset + offset²
        # For small offset, this is approximately 2 * sqrt_n * offset
        if offset == 0
            sieve[i] = 0.0  # Q(sqrt_n) ≈ 0 or small
        else
            sieve[i] = log_sqrt_n + log(2.0) + log(Float64(offset))
        end
    end

    return sieve, sqrt_n
end

"""
    sieve_with_prime!(sieve::Vector{Float64}, fb::QSFactorBase, idx::Int,
                      n::Integer, sqrt_n::Integer, M::Int)

Sieve the array with a prime from the factor base, subtracting log(p) at divisible positions.
"""
function sieve_with_prime!(sieve::Vector{Float64}, fb::QSFactorBase, idx::Int,
                           n::Integer, sqrt_n::Integer, M::Int)
    p = fb.primes[idx]
    log_p = fb.log_p[idx]
    sqrt_n_p = fb.sqrt_n_mod_p[idx]

    # For p=2, handle specially
    if p == 2
        # Q(x) = x² - n is even when x has same parity as n
        offset = mod(sqrt_n, 2) == mod(n, 2) ? 1 : 2
        for i in offset:2:(M + 1)
            sieve[i] -= log_p
        end
        return
    end

    # For odd primes, find starting positions
    # Q(x) ≡ 0 (mod p) when x ≡ ±√n (mod p)
    sqrt_n_mod = mod(sqrt_n, p)

    # Two roots: sqrt_n_p and p - sqrt_n_p
    for root in (sqrt_n_p, p - sqrt_n_p)
        # Find first offset where sqrt_n + offset ≡ root (mod p)
        offset = mod(root - sqrt_n_mod, p)
        pos = offset + 1  # +1 for 1-based indexing
        while pos <= M + 1
            sieve[pos] -= log_p
            pos += p
        end
    end
end

"""
    collect_smooth_candidates(sieve::Vector{Float64}, threshold::Float64) -> Vector{Int}

Collect indices where the sieve value is below threshold, indicating potential smooth numbers.
"""
function collect_smooth_candidates(sieve::Vector{Float64}, threshold::Float64)
    candidates = Int[]
    for i in eachindex(sieve)
        if sieve[i] < threshold
            push!(candidates, i)
        end
    end
    return candidates
end

#=============================================================================
  Smooth Number Verification and Factoring
=============================================================================#

"""
    factor_over_base!(exponents::Vector{Int}, q::Integer, fb::QSFactorBase) -> Bool

Attempt to factor q completely over the factor base, storing exponents in the provided buffer.
Returns true if successful, false if q has factors outside the base.
The exponents buffer is modified in place and zeroed at the start.
"""
function factor_over_base!(exponents::Vector{Int}, q::T, fb::QSFactorBase) where T<:Integer
    fill!(exponents, 0)

    # Handle sign
    if q < 0
        q = -q
    end

    q == 0 && return false

    # Trial divide by each prime in factor base
    for (i, p) in enumerate(fb.primes)
        while q > 1 && mod(q, p) == 0
            q = div(q, p)
            exponents[i] += 1
        end
        q == 1 && break
    end

    # If q != 1, it has factors outside the factor base
    return q == 1
end

"""
    factor_over_base(q::Integer, fb::QSFactorBase) -> Union{Vector{Int}, Nothing}

Attempt to factor q completely over the factor base.
Returns the exponent vector if successful, nothing if q has factors outside the base.
"""
function factor_over_base(q::T, fb::QSFactorBase) where T<:Integer
    exponents = zeros(Int, length(fb.primes))
    if factor_over_base!(exponents, q, fb)
        return exponents
    end
    return nothing
end

"""
    verify_and_collect_smooth!(smooth_numbers::Vector{QSSmoothNumber{T}},
                                candidates::Vector{Int}, n::T, start_x::T,
                                fb::QSFactorBase,
                                seen_x::Set{T}=Set{T}()) where T<:Integer

Verify smooth candidates and collect confirmed smooth numbers.
Avoids duplicates by tracking seen x values.
"""
function verify_and_collect_smooth!(smooth_numbers::Vector{QSSmoothNumber{T}},
                                    candidates::Vector{Int}, n::T, start_x::T,
                                    fb::QSFactorBase,
                                    seen_x::Set{T}=Set{T}()) where T<:Integer
    # Pre-allocate exponent buffer for efficiency
    exponents_buffer = zeros(Int, length(fb.primes))

    for idx in candidates
        x = start_x + idx - 1

        # Skip duplicates
        x in seen_x && continue

        q = x * x - n

        if factor_over_base!(exponents_buffer, q, fb)
            # Copy exponents since buffer will be reused
            push!(smooth_numbers, QSSmoothNumber{T}(x, q, copy(exponents_buffer)))
            push!(seen_x, x)
        end
    end
end

#=============================================================================
  Linear Algebra over GF(2)
=============================================================================#

"""
    build_exponent_matrix(smooth_numbers::Vector{QSSmoothNumber{T}},
                          num_primes::Int) -> QSMatrix where T

Build the exponent parity matrix from smooth numbers.
"""
function build_exponent_matrix(smooth_numbers::Vector{QSSmoothNumber{T}},
                               num_primes::Int) where T
    rows = BitVector[]
    indices = Int[]

    for (i, sn) in enumerate(smooth_numbers)
        # Convert exponents to parities (mod 2)
        parity = BitVector(mod.(sn.exponents, 2) .== 1)

        # Handle the sign bit: if q < 0, we need to track the -1 factor
        # We'll prepend a bit for the sign
        sign_bit = sn.q < 0
        row = vcat(BitVector([sign_bit]), parity)

        push!(rows, row)
        push!(indices, i)
    end

    return QSMatrix(rows, indices, num_primes + 1)  # +1 for sign column
end

"""
    gaussian_elimination_gf2!(matrix::QSMatrix) -> Vector{Vector{Int}}

Perform Gaussian elimination over GF(2) and find null space vectors.
Returns a list of null space vectors, each represented as indices of rows that sum to zero.
"""
function gaussian_elimination_gf2!(matrix::QSMatrix)
    rows = matrix.rows
    n_rows = length(rows)
    n_cols = matrix.num_primes

    n_rows == 0 && return Vector{Int}[]

    # Track which rows contribute to each position
    row_history = [Set{Int}([i]) for i in 1:n_rows]

    # Track pivot positions
    pivot_row = zeros(Int, n_cols)

    # Forward elimination
    current_row = 1
    for col in 1:n_cols
        # Find pivot
        pivot_found = false
        for row in current_row:n_rows
            if rows[row][col]
                # Swap rows if needed
                if row != current_row
                    rows[row], rows[current_row] = rows[current_row], rows[row]
                    row_history[row], row_history[current_row] = row_history[current_row], row_history[row]
                end
                pivot_found = true
                break
            end
        end

        !pivot_found && continue

        pivot_row[col] = current_row

        # Eliminate this column in other rows
        for row in 1:n_rows
            if row != current_row && rows[row][col]
                rows[row] = rows[row] .⊻ rows[current_row]
                row_history[row] = symdiff(row_history[row], row_history[current_row])
            end
        end

        current_row += 1
        current_row > n_rows && break
    end

    # Find null space vectors (rows that became zero)
    null_vectors = Vector{Int}[]
    for i in 1:n_rows
        if all(==(false), rows[i])
            push!(null_vectors, sort!(collect(row_history[i])))
        end
    end

    return null_vectors
end

#=============================================================================
  Factor Extraction
=============================================================================#

"""
    extract_factor(n::T, null_vector::Vector{Int},
                   smooth_numbers::Vector{QSSmoothNumber{T}}) -> Union{T, Nothing} where T

Attempt to extract a non-trivial factor using a null space vector.
"""
function extract_factor(n::T, null_vector::Vector{Int},
                        smooth_numbers::Vector{QSSmoothNumber{T}}) where T<:Integer
    length(null_vector) == 0 && return nothing

    # Compute X = product of x values
    X = one(T)
    for idx in null_vector
        X = mod(X * smooth_numbers[idx].x, n)
    end

    # Compute Y² = product of Q(x) values, then take square root
    # First sum the exponents
    total_exponents = zeros(Int, length(smooth_numbers[1].exponents))
    total_sign_exp = 0
    for idx in null_vector
        total_exponents .+= smooth_numbers[idx].exponents
        if smooth_numbers[idx].q < 0
            total_sign_exp += 1
        end
    end

    # All exponents should be even (that's what null space means)
    # If sign exponent is odd, we have a problem
    isodd(total_sign_exp) && return nothing

    # Compute Y from the half-exponents
    # We need access to the factor base primes, so we'll compute Y differently:
    # Y² = ∏ Q(x_i), so Y = √(∏ Q(x_i))
    # Since exponents are even, we can compute this

    # Compute the product of Q values (use BigInt to avoid overflow)
    prod_q = one(BigInt)
    for idx in null_vector
        q = smooth_numbers[idx].q
        prod_q *= q
    end

    # prod_q should be a perfect square
    prod_q < 0 && return nothing  # Shouldn't happen if total_sign_exp is even

    Y_big = isqrt(prod_q)
    Y_big * Y_big != prod_q && return nothing  # Not a perfect square (shouldn't happen)

    Y = T(mod(Y_big, n))

    # Try gcd(X - Y, n) and gcd(X + Y, n)
    g1 = gcd(mod(X - Y + n, n), n)
    if g1 != 1 && g1 != n
        return g1
    end

    g2 = gcd(mod(X + Y, n), n)
    if g2 != 1 && g2 != n
        return g2
    end

    return nothing
end

#=============================================================================
  Main Algorithm
=============================================================================#

"""
    quadratic_sieve_factor(n::T) -> T where T<:Integer

Find a non-trivial factor of n using the Quadratic Sieve algorithm.

This algorithm is effective for factoring integers in the range of 40-70 decimal digits
with balanced prime factors (factors of similar size).

# Arguments
- `n`: A composite integer to factor. Must be > 1, odd, and not a prime.

# Returns
A non-trivial factor p of n where 1 < p < n.
Note: The returned factor is not guaranteed to be prime.

# Algorithm
The quadratic sieve works by:
1. Building a factor base of small primes
2. Sieving to find "smooth" numbers Q(x) = x² - n
3. Using linear algebra over GF(2) to find products of Q(x) that are perfect squares
4. Extracting factors using the congruence of squares

# References
- Pomerance, Carl. "The quadratic sieve factoring algorithm." (1984)
- https://en.wikipedia.org/wiki/Quadratic_sieve

# Example
```jldoctest
julia> using Primes

julia> # Factor a 20-digit semiprime
julia> p, q = nextprime(big(10)^9), nextprime(big(10)^9 + 1000);

julia> n = p * q;

julia> f = Primes.quadratic_sieve_factor(n);

julia> n % f == 0
true

julia> 1 < f < n
true
```
"""
function quadratic_sieve_factor(n::T) where T<:Integer
    # Input validation
    n > 1 || throw(ArgumentError("n must be > 1"))
    !iseven(n) || throw(ArgumentError("n must be odd"))
    !isprime(n) || throw(ArgumentError("n must be composite"))

    # Convert to BigInt for safety in intermediate calculations
    n_big = BigInt(n)

    # Compute parameters - use conservative values to avoid memory issues
    num_digits = ndigits(n_big)

    # Adaptive parameter selection based on number size
    # B = factor base bound (number of primes)
    # M = sieve interval size (larger = more smooth numbers but more memory)
    if num_digits <= 15
        B = 200
        M = 5_000
    elseif num_digits <= 20
        B = 300
        M = 20_000
    elseif num_digits <= 25
        B = 500
        M = 100_000
    elseif num_digits <= 30
        B = 1_000
        M = 500_000
    elseif num_digits <= 35
        B = 2_000
        M = 1_000_000
    elseif num_digits <= 45
        B = 10_000
        M = 2_000_000
    elseif num_digits <= 55
        B = 50_000
        M = 5_000_000
    else
        B = 200_000
        M = 10_000_000
    end

    # Build factor base
    fb = build_factor_base(n_big, B)
    num_primes = length(fb.primes)

    num_primes == 0 && error("Failed to build factor base")

    # We need at least num_primes + 1 smooth numbers
    target_smooth = num_primes + 10  # Some buffer

    sqrt_n = isqrt(n_big)

    # Sieving
    smooth_numbers = QSSmoothNumber{BigInt}[]
    seen_x = Set{BigInt}()  # Track seen x values to avoid duplicates

    # Sieve threshold: numbers with log(Q(x)) below this after sieving are likely smooth
    threshold = log(Float64(fb.primes[end])) + 5.0

    # Expand sieving interval if needed
    sieve_attempts = 0
    max_attempts = 20
    current_M = M

    while length(smooth_numbers) < target_smooth && sieve_attempts < max_attempts
        sieve_attempts += 1

        # Initialize sieve
        sieve, start_x = init_sieve_array(n_big, sqrt_n, current_M)

        # Sieve with each prime
        for i in eachindex(fb.primes)
            sieve_with_prime!(sieve, fb, i, n_big, sqrt_n, current_M)
        end

        # Collect candidates
        candidates = collect_smooth_candidates(sieve, threshold)

        # Verify and collect smooth numbers (pass seen_x to avoid duplicates)
        verify_and_collect_smooth!(smooth_numbers, candidates, n_big, start_x, fb, seen_x)

        # If not enough, expand interval and be more lenient
        if length(smooth_numbers) < target_smooth
            current_M = min(current_M * 2, 50_000_000)
            threshold += 2.0  # Be more lenient
        end

        # Yield to allow interruption
        yield()
    end

    length(smooth_numbers) == 0 && error("Failed to find smooth numbers after $max_attempts attempts")

    # Build matrix and find null space
    matrix = build_exponent_matrix(smooth_numbers, num_primes)
    null_vectors = gaussian_elimination_gf2!(matrix)

    length(null_vectors) == 0 && error("Failed to find null space vectors")

    # Try each null vector to extract a factor
    for null_vec in null_vectors
        factor_result = extract_factor(n_big, null_vec, smooth_numbers)
        if factor_result !== nothing
            return T(factor_result)
        end
        yield()
    end

    error("Failed to extract factor from null vectors")
end
