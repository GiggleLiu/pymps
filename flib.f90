!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Fortran version block finding for csr type matrix.
!Complexity: sum over p**2, p is the collection of block size.
!Author: Leo
!Data: Oct. 8. 2015
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!merge 1-d MPOs into one.
!
!Parameters
!-------------------------
!ol: 4d array.
!    the decoupled tensor form.
!
!Return:
!--------------------------
!res: 4d array,
subroutine fmerge_mpo(ol,hndim,bn,res)
    implicit none
    integer,intent(in) :: bn,hndim
    complex*16,intent(in) :: ol(bn,hndim,hndim,1)
    complex*16,intent(out) :: res(bn,hndim,hndim,bn)
    integer :: j
    
    !f2py intent(in) :: ol,hndim,bn
    !f2py intent(out) :: res

    res=0
    do j=1,bn
        res(j,:,:,j)=ol(j,:,:,1)
    enddo
end subroutine fmerge_mpo
