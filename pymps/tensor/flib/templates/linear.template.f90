!orders: batch_dim, feature_dim_out/in
module lib
    contains
    _gen_index_table(nzblocks, key_axes)

    {%for dtype in dtype_list -%}
    {%if dtype == "complex*16"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "dcmplx(1D0,0D0)", "dcmplx(0D0,0D0)", "z", True -%}
    {%elif dtype == "complex*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex ="cmplx(1.0,0.0)", "cmplx(0.0,0.0)", "c", True -%}
    {%elif dtype == "real*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1D0", "0D0", "d", False -%}
    {%elif dtype == "real*4"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1.0", "0.0", "s", False -%}
    {%endif -%}
    subroutine btdot_{{dtype_token}}(t1, t2, inner1, inner2, outer1, outer2, nr1, nr2, signs1, signs2, per, outarr)
    ! output array
    out_shape = [tensor1.shape[i] for i in outer1]+[tensor2.shape[i] for i in outer2]
    out_arr = np.zeros(out_shape, dtype=np.find_common_type((tensor1.dtype, tensor2.dtype), ()))
    out_arr = Tensor(out_arr, labels=[tensor1.labels[i] for i in outer1]+[tensor2.labels[i] for i in outer2])

    ! get non-zero blocks for tensor1 and tensor2
    nz_table1 = _gen_index_table(zero_flux_blocks([l.bm.qns for l in tensor1.labels], signs1, bmg), key_axes=inner1)
    nz_table2 = _gen_index_table(zero_flux_blocks([l.bm.qns for l in tensor2.labels], signs2, bmg), key_axes=inner2)

    for k, l1 in nz_table1.items():
        if k in nz_table2:
            l2 = nz_table2[k]
            for b1, o1 in l1:
                for b2, o2 in l2:
                    out_b = o1+o2
                    bdata = np.tensordot(tensor1.get_block(b1), tensor2.get_block(b2), axes=(inner1, inner2))
                    out_arr.set_block(out_b, bdata)
    end subroutine
end module lib
