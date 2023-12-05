# abacus中的集合通信

函数 mpi_check_consistency

abacus_read_from(), read feasign path
    mpi_check_consistency(&path, 1);
    mpi_rank() == 1, fs_list(path)
    the channel returned callback function call mpi_barrier()

abacus_write_into()
    mpi_check_constistency(&path, 1)
    mpi_check_constistency(&pattern, 1), pattern is file name

begin_day()
    first and last: mpi_barrier()

end_day()
    first and last: mpi_barrier()
    mpi_allreduce() sum/min/max consumend time

shrink_table()
    first and last: mpi_barrier()

begin_pass()
    first: mpi_barrier()

end_pass()
    first: mpi_barrier()
    mpi_allreaduce consumed_time

load_model()
    after load: mpi_barrier()

save_model()
    after save: mpi_barrier()

save_patch_model()
    after save: mpi_barrier()

save_cache_meta_donefile()
    mpi_allreduce(_data_size)

save_delta_model()
    after save cahce: mpi_barrier()

