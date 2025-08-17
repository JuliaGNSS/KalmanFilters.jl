for (gels, gesv, getrs, getri, elty) in (
    (:dgels_, :dgesv_, :dgetrs_, :dgetri_, :Float64),
    (:sgels_, :sgesv_, :sgetrs_, :sgetri_, :Float32),
    (:zgels_, :zgesv_, :zgetrs_, :zgetri_, :ComplexF64),
    (:cgels_, :cgesv_, :cgetrs_, :cgetri_, :ComplexF32),
)
    @eval begin
        #      SUBROUTINE DGELS( TRANS, M, N, NRHS, A, LDA, B, LDB, WORK, LWORK,INFO)
        # *     .. Scalar Arguments ..
        #       CHARACTER          TRANS
        #       INTEGER            INFO, LDA, LDB, LWORK, M, N, NRHS
        function mygels!(
            res::AbstractMatrix{$elty},
            A::AbstractMatrix{$elty},
            B::AbstractVecOrMat{$elty},
            work::Vector{$elty},
        )
            trans = 'N'
            Base.require_one_based_indexing(A, B)
            LAPACK.chktrans(trans)
            chkstride1(A, B)
            btrn = trans == 'T'
            m, n = size(A)
            if size(B, 1) != (btrn ? n : m)
                throw(
                    DimensionMismatch(
                        "matrix A has dimensions ($m,$n), transposed: $btrn, but leading dimension of B is $(size(B,1))",
                    ),
                )
            end
            info = Ref{BlasInt}()
            lwork = BlasInt(length(work))
            ccall(
                (@blasfunc($gels), liblapack),
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{BlasInt},
                    Clong,
                ),
                (btrn ? 'T' : 'N'),
                m,
                n,
                size(B, 2),
                A,
                max(1, stride(A, 2)),
                B,
                max(1, stride(B, 2)),
                work,
                lwork,
                info,
                1,
            )
            LAPACK.chklapackerror(info[])
            k = min(m, n)
            res .= @view(A[1:k, 1:k])
            m < n ? tril!(res) : triu!(res)
            res
        end

        function calc_gels_working_size(
            A::AbstractMatrix{$elty},
            B::AbstractVecOrMat{$elty},
        )
            trans = 'N'
            Base.require_one_based_indexing(A, B)
            LAPACK.chktrans(trans)
            chkstride1(A, B)
            btrn = trans == 'T'
            m, n = size(A)
            if size(B, 1) != (btrn ? n : m)
                throw(
                    DimensionMismatch(
                        "matrix A has dimensions ($m,$n), transposed: $btrn, but leading dimension of B is $(size(B,1))",
                    ),
                )
            end
            info = Ref{BlasInt}()
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            ccall(
                (@blasfunc($gels), liblapack),
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{BlasInt},
                    Clong,
                ),
                (btrn ? 'T' : 'N'),
                m,
                n,
                size(B, 2),
                A,
                max(1, stride(A, 2)),
                B,
                max(1, stride(B, 2)),
                work,
                lwork,
                info,
                1,
            )
            LAPACK.chklapackerror(info[])
            BlasInt(real(work[1]))
        end
    end
end

"""
    mygels!(trans, A, B) -> (F)

Solves the linear equation `A * X = B`, `transpose(A) * X = B`, or `adjoint(A) * X = B` using
a QR or LQ factorization. Modifies the matrix/vector `B` in place with the
solution. `A` is overwritten with its `QR` or `LQ` factorization. `trans`
may be one of `N` (no modification), `T` (transpose), or `C` (conjugate
transpose). `gels!` searches for the minimum norm/least squares solution.
`A` may be under or over determined. The solution is returned in `B`.
"""
mygels!(trans::AbstractChar, A::AbstractMatrix, B::AbstractVecOrMat, work::Vector)

"""
    mygels!(trans, A, B) -> (working_size)

Calculates working size for mygels!
"""
calc_gels_working_size(trans::AbstractChar, A::AbstractMatrix, B::AbstractVecOrMat)
