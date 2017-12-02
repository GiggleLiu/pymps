!orders: batch_dim, feature_dim_out/in
module lib
    contains
    subroutine create_table(contract_qns, left_labels, right_labels, )
        implicit none
        integer,intent(in): contract_qns(num_clegs,num_)
    end subroutine create_table

    {%for dtype in dtype_list -%}
    {%if dtype == "complex*16"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "dcmplx(1D0,0D0)", "dcmplx(0D0,0D0)", "z", True -%}
    {%elif dtype == "complex*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex ="cmplx(1.0,0.0)", "cmplx(0.0,0.0)", "c", True -%}
    {%elif dtype == "real*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1D0", "0D0", "d", False -%}
    {%elif dtype == "real*4"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1.0", "0.0", "s", False -%}
    {%endif -%}
    subroutine beinsum_{{dtype_token}}(x, y):
        implicit none
        integer,intent(in) :: num_batch, nfi, nfo
        {{dtype}},intent(in) :: x(num_batch, nfi), weight(nfo, nfi), bias(nfo)
        {%if version == "masked"%}logical,intent(in) :: mask(nfo, nfi){%endif%}
        {{dtype}},intent(out) :: y(num_batch, nfo)
        {{dtype}},parameter :: one={{dtype_one}}
        integer :: i

        do i=1,nfo
            y(:,i)=bias(i)
        enddo

        call {{dtype_token}}gemm('N', 'T', num_batch, nfo, nfi, one, x, num_batch,&
            weight, nfo, one, y, num_batch)
    end subroutine forward_{{version}}{{dtype_token}}
    {%endfor -%}
end module lib
