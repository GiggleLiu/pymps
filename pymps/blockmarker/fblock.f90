!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Fortran version block finding for csr type matrix.
!Complexity: sum over p**2, p is the collection of block size.
!Author: Leo
!Data: Oct. 8. 2015
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!Get the permutation order that will turn a symmetrix/hermitian matrix into block diagonal form.
!
!indices:
!    the y indices.
!indptr:
!    the row indicator in c(b)sr_matrix form.
!
!return:
!    A BlockIndicator instance.
subroutine fblockizeh(indices,indptr,pmat,n,nitem,nr,nblock)
    implicit none
    integer,intent(in) :: n,nitem
    integer,intent(in) :: indptr(n+1),indices(nitem)
    integer,intent(out),dimension(n) :: pmat
    integer,intent(out) :: nr(n+1),nblock
    integer :: i,j,psize,lsize
    logical,dimension(n) :: mask
    integer,dimension(n) :: temp
    
    !f2py intent(in) :: n,nitem,indptr,indices
    !f2py intent(out) :: pmat,nr,nblock

    mask=.false.
    psize=0
    nblock=0
    nr(1)=0
    do i=0,n-1
        if(.not. mask(i+1)) then
            lsize=0
            call enclosure(temp,i,indices,indptr,n,nitem,lsize)
            do j=1,lsize
                mask(temp(j)+1)=.true.
            enddo
            pmat(psize+1:psize+lsize)=temp(1:lsize)
            psize=psize+lsize
            nblock=nblock+1
            nr(nblock+1)=psize
        endif
    enddo
end subroutine fblockizeh

!Group up qns and get the size of each qn, the scalar version.
!
!Parameters:
!   qns_sorted, the sorted qns in 1D array.
!   N, the size of qns_sorted.
!
!Return:
!   nnr, the size of each block.
!   nn, the number of blocks.
!   gqns, the qns of groups.
subroutine fgroup_iqns(qns_sorted,n,nnr,gqns,nn)
    implicit none
    integer,intent(in) :: n
    integer,intent(in) :: qns_sorted(n)
    integer,intent(out) :: nn,nnr(n),gqns(n)
    integer :: i,temp,qn_pre,qn
    
    !f2py intent(in) :: qns_sorted,n
    !f2py intent(out) :: nnr,gqns,nn

    qn_pre=-10000000
    temp=0
    nn=0
    do i=1,n
        qn=qns_sorted(i)
        if(i==1 .or. qn/=qn_pre) then
            nn=nn+1
            gqns(nn)=qn
            if(i/=1) then
                nnr(nn-1)=temp
            endif
            temp=1
            qn_pre=qn
        else
            temp=temp+1
        endif
    enddo
    nnr(nn)=temp
end subroutine fgroup_iqns

!Group up qns and get the size of each qn, the float version.
!
!Parameters:
!   qns_sorted, the sorted qns in 1D array.
!   N, the size of qns_sorted.
!
!Return:
!   nnr, the size of each block.
!   nn, the number of blocks.
!   gqns, the qns of groups.
subroutine fgroup_dqns(qns_sorted,n,nnr,gqns,nn)
    implicit none
    integer,intent(in) :: n
    real*8,intent(in) :: qns_sorted(n)
    integer,intent(out) :: nn,nnr(n)
    real*8,intent(out) :: gqns(n)
    integer :: i,temp
    real*8 :: qn_pre,qn
    
    !f2py intent(in) :: qns_sorted,n
    !f2py intent(out) :: nnr,gqns,nn

    qn_pre=-1D8
    temp=0
    nn=0
    do i=1,n
        qn=qns_sorted(i)
        if(i==1 .or. abs(qn-qn_pre)>1D-8) then
            nn=nn+1
            gqns(nn)=qn
            if(i/=1) then
                nnr(nn-1)=temp
            endif
            temp=1
            qn_pre=qn
        else
            temp=temp+1
        endif
    enddo
    nnr(nn)=temp
end subroutine fgroup_dqns

!Group up qns and get the size of each qn, the array version.
!
!Parameters:
!   qns_sorted, the sorted qns in 1D array.
!   N, the size of qns_sorted.
!
!Return:
!   nnr, the size of each block.
!   nn, the number of blocks.
!   gqns, the qns of groups.
subroutine fgroup_aqns(qns_sorted,n,m,nnr,gqns,nn)
    implicit none
    integer,intent(in) :: n,m
    real*8,intent(in) :: qns_sorted(n,m)
    integer,intent(out) :: nn,nnr(n)
    real*8,intent(out) :: gqns(n,m)
    integer :: i,temp
    real*8 :: qn_pre(m),qn(m)
    
    !f2py intent(in) :: qns_sorted,n,m
    !f2py intent(out) :: nnr,gqns,nn

    qn_pre=(/-1D8,-1D8/)
    temp=0
    nn=0
    do i=1,n
        qn=qns_sorted(i,:)
        if(i==1 .or. sum((qn-qn_pre)**2)>1D-8) then
            nn=nn+1
            gqns(nn,:)=qn(:)
            if(i/=1) then
                nnr(nn-1)=temp
            endif
            temp=1
            qn_pre=qn
        else
            temp=temp+1
        endif
    enddo
    nnr(nn)=temp
end subroutine fgroup_aqns

recursive subroutine enclosure(l,ii,indices,indptr,n,nitem,lsize)
    implicit none
    integer,intent(in) :: ii,n,nitem
    integer,intent(in) :: indptr(n+1),indices(nitem)
    integer,intent(inout) :: lsize
    integer,intent(inout),dimension(n) :: l
    integer :: k,jj,njj,i
    integer,dimension(:),allocatable :: jjs
    logical :: jjinl
    lsize=lsize+1
    l(lsize)=ii
    njj=indptr(ii+2)-indptr(ii+1)
    allocate(jjs(njj))
    jjs=indices(indptr(ii+1)+1:indptr(ii+2))
    do i=1,njj
        jj=jjs(i)
        jjinl=.false.
        do k=1,lsize
            if(l(k)==jj) then
                jjinl=.true.
                exit
            endif
        enddo
        if(.not. jjinl) then
            call enclosure(l,jj,indices,indptr,n,nitem,lsize)
        endif
    enddo
end subroutine enclosure
