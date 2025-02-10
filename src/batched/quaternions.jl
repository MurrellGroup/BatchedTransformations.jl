using PaddedViews

# parameterized by a 4xN array of unit quaternions
struct QuaternionRotation{T,A<:AbstractArray{T}} <: AbstractLinear{Rotation,T}
    values::A
end

const QUAT_FIELDS = (:w, :x, :y, :z)
const QUAT_INDEX = NamedTuple(QUAT_FIELDS .=> 1:4)

function Base.getproperty(qr::QuaternionRotation, p::Symbol)
    return if p in QUAT_FIELDS
        i = QUAT_INDEX[p] - 1 + first(axes(qr.values, 1))
        @view qr.values[i:i, ..]
    else
        getfield(qr, p)
    end
end

linear(qr::QuaternionRotation) = qr
translation(::QuaternionRotation) = Identity()

batchsize(qr::QuaternionRotation) = size(qr.values)[2:end]
batchrepeat(qr::QuaternionRotation, outer...) = QuaternionRotation(repeat(qr.values, 1, outer...))
batchreshape(qr::QuaternionRotation, dims...) = QuaternionRotation(reshape(qr.values, 4, dims...))
batchunsqueeze(qr::QuaternionRotation; dims::Int) = QuaternionRotation(unsqueeze(qr.values, dims=1+dims))

Base.values(qr::QuaternionRotation) = qr.values

norms(A::AbstractArray{<:Number}; dims=1) = sqrt.(sum(abs2, A; dims))

# conjugate since we assume unit quaternions
function Base.inv(qr::QuaternionRotation)
    Q = values(qr)
    @views a, bcd = Q[1:1, ..], Q[2:4, ..]
    return QuaternionRotation([a; -bcd])
end

function transform(q::QuaternionRotation, p::QuaternionRotation)
    qw, qx, qy, qz = q.w, q.x, q.y, q.z
    pw, px, py, pz = p.w, p.x, p.y, p.z
    return QuaternionRotation([
        qw .* pw - qx .* px - qy .* py - qz .* pz
        qw .* px + qx .* pw + qy .* pz - qz .* py
        qw .* py - qx .* pz + qy .* pw + qz .* px
        qw .* pz + qx .* py - qy .* px + qz .* pw
    ])
end

function transform(q::QuaternionRotation, p::AbstractArray{<:Number})
    size(p, 1) == 3 || throw(ArgumentError("p must have shape 3xB..."))
    p = QuaternionRotation(PaddedView(0, p, (0:size(p, 1), axes(p)[2:end]...)))
    q̅ = inv(q)
    return @view values((q * p) * q̅)[2:4, ..]
end

function Base.convert(::Type{QuaternionRotation}, r::Rotation)
    R = values(r)
    size(R)[1:2] == (3,3) || throw(ArgumentError("Rotation matrix batch must have shape 3x3xB..."))
    # 1x1xB
    @views r11, r12, r13 = R[1:1, 1:1, ..], R[1:1, 2:2, ..], R[1:1, 3:3, ..]
    @views r21, r22, r23 = R[2:2, 1:1, ..], R[2:2, 2:2, ..], R[2:2, 3:3, ..]
    @views r31, r32, r33 = R[3:3, 1:1, ..], R[3:3, 2:2, ..], R[3:3, 3:3, ..]

    # 4x1xB...
    q0 = [1 .+ r11 + r22 + r33
          r32 - r23
          r13 - r31
          r21 - r12]
    q1 = [r32 - r23
          1 .+ r11 - r22 - r33
          r12 + r21
          r13 + r31]
    q2 = [r13 - r31
          r12 + r21
          1 .- r11 + r22 - r33
          r23 + r32]
    q3 = [r21 - r12
          r13 + r31
          r23 + r32
          1 .- r11 - r22 + r33]

    # 4x4xB...
    qs = hcat(q0, q1, q2, q3)

    # 1x4xB..., norm of each quaternion
    q_norms = norms(qs, dims=1)

    exp_norms = exp.(q_norms)

    # 1x4xB..., norm weights
    weights = exp_norms ./ sum(exp_norms, dims=3:ndims(exp_norms))

    # batched matmul, 4xB...
    Q = dropdims(batched_mul(qs, unsqueeze(dropdims(weights, dims=1), dims=2)), dims=2)
    Q_normalized = Q ./ norms(Q, dims=1)

    return QuaternionRotation(Q_normalized)
end

# From Algorithm 23 of AlphaFold 2 supplementary information
function Base.convert(::Type{Rotation}, qr::QuaternionRotation{T}) where T<:Number
    Q = values(qr)

    @views a = unsqueeze(Q[1:1, ..], dims=1)
    @views b = unsqueeze(Q[2:2, ..], dims=1)
    @views c = unsqueeze(Q[3:3, ..], dims=1)
    @views d = unsqueeze(Q[4:4, ..], dims=1)

    ab = a .* b
    ac = a .* c
    ad = a .* d
    bb = b .^ 2
    bc = b .* c
    bd = b .* d
    cc = c .^ 2
    cd = c .* d
    dd = d .^ 2

    h = T(1 // 2)
    return Rotation(2 * [
        h .- (cc + dd)        bc - ad         bd + ac
              bc + ad   h .- (bb + dd)        cd - ab
              bd - ac         cd + ab   h .- (bb + cc)
    ])
end

"""
    imaginary_to_quaternion_rotations(bcd::AbstractArray, a²=1)

Convert a batch of imaginary vectors represented as an array of size 3xB to a batch of unit quaternions.

Taken from Algorithm 23 of AlphaFold 2 supplementary information.
"""
function imaginary_to_quaternion_rotations(bcd::AbstractArray{T}, a²::T=T(1)) where T<:Number
    size(bcd, 1) == 3 || throw(ArgumentError("bcd must have shape 3xB..."))
    norms = sqrt.(a² .+ sum(abs2, bcd, dims=1))
    a = sqrt(a²)
    return [a ./ norms; bcd ./ norms]
end
