#include <tbb/task_scheduler_observer.h>
#include <tbb/atomic.h>
#include <tbb/task_arena.h>
#include <hwloc.h>

namespace numa{

class PinningObserver : public tbb::task_scheduler_observer {
    hwloc_topology_t topo;
    hwloc_obj_t numa_node;
    int numa_id;
    int numa_nodes;
    tbb::atomic<int> thds_per_node;
    tbb::atomic<int> masters_that_entered;
    tbb::atomic<int> workers_that_entered;
    tbb::atomic<int> threads_pinned;
public:
    PinningObserver(tbb::task_arena& arena, hwloc_topology_t& _topo, int _numa_id,
                    int _thds_per_node) : task_scheduler_observer(arena), topo(_topo),
                    numa_id(_numa_id), thds_per_node(_thds_per_node)
    {
        numa_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
        numa_node = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NUMANODE, numa_id);
        masters_that_entered = 0;
        workers_that_entered = 0;
        threads_pinned = 0;
        observe(true);
    }                 

    void on_scheduler_entry(bool is_worker)
    {
        if (is_worker) ++workers_that_entered;
        else ++masters_that_entered;
        if(--thds_per_node > 0)
        {
            int err = hwloc_set_cpubind(topo, numa_node->cpuset, HWLOC_CPUBIND_THREAD);
            assert(!err);
            threads_pinned++;
        }
    }
};

void alloc_mem_per_node(hwloc_topology_t topo, float** data_in, float** data_out, size_t size, size_t width){
	int num_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
	for (int i = 0; i < num_nodes; i++){
		hwloc_obj_t numa_node = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NUMANODE, i);

		data_in[i] = (float *) hwloc_alloc_membind(topo, size * sizeof(float), numa_node->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
        float *A = data_in[i];

        float *B = nullptr;
        if(i == 0 || i == 3){
            data_out[i] = (float *) hwloc_alloc_membind(topo, size * sizeof(float), numa_node->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
            B = data_out[i];
        }
        else if (i == 1){
            data_out[2] = (float *) hwloc_alloc_membind(topo, size * sizeof(float), numa_node->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
            B = data_out[2];
        }
        else if (i == 2){
            data_out[1] = (float *) hwloc_alloc_membind(topo, size * sizeof(float), numa_node->nodeset, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
            B = data_out[1];
        }

        size_t count = 0;
        for (size_t j = 0; j < size; j++){
            B[j] = -1.0;
            if (i == 0) A[j] = (float) j + count * width / 2;
            else if (i == 1) A[j] = (float) j + (count + 1) * width / 2;
            else if (i == 2) A[j] = (float) j + 2 * size + count * width / 2;
            else if (i == 3) A[j] = (float) j + 2 * size + (count + 1) * width / 2;
            if(!((j+1)%(width/2))) count++;
        } 
	}
}

void alloc_thr_per_node(hwloc_topology_t topo, float** data_in, float** data_out, size_t size, int thds_per_node, size_t width, int gs){
    size_t num_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
    std::vector<std::thread> vth;
    size_t sub_width = width / 2;
    for(int i = 0; i < num_nodes; i++){
        vth.push_back(std::thread{
            [=, &topo](){
                hwloc_obj_t numa_node = hwloc_get_obj_by_type(topo, HWLOC_OBJ_NUMANODE, i);
                int err = hwloc_set_cpubind(topo, numa_node->cpuset, HWLOC_CPUBIND_THREAD);

                assert(!err);

                tbb::task_arena numa_arena{thds_per_node};
                PinningObserver p{numa_arena, topo, i, thds_per_node};

                float *A = data_in[i];
                float *B = nullptr;
                if (i == 0 || i == 3) B = data_out[i];
                else if (i == 1) B = data_out[2];
                else if (i == 2) B = data_out[1];

                numa_arena.execute([&](){
                    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, sub_width, gs, 0, sub_width, gs),
                    [&](const tbb::blocked_range2d<size_t>& r){
                        for(size_t x = r.cols().begin(); x + 8 <= r.cols().end(); x+=8){
                            for(size_t y = r.rows().begin(); y + 8 <= r.rows().end(); y+=8){
                                //B[x * sub_width + y] = A[y * sub_width + x];
                                tran(&A[y * sub_width + x], &B[x * sub_width + y], sub_width, sub_width);
                            }
                        }
                    },tbb::simple_partitioner());
                });
            }
        });
    }
    for (auto &th: vth) th.join();

}
}