2026.6.10 Version = 0.0.2
1.update analyse.PDF -> ratio_3pt, To obtain the matrix elements for the three-point functions that satisfy different initial and final states
2.remove analyse.get_mpi_info
3.Increase constant.Mom_cross_sigma about the P cross sigma function
4.Increase eigvectors.vertex.sink2src from sink vertex to get the source vertex

2026.6.16 Version = 0.0.3
1.fix bug of "ratio_3pt", about the movement of time-based indicators.
2.Increase analyse.mean_over_array_of_list it same like sum_over_array_of_list, just from sum to mean.

3.Modify eigvectors.vertex.VdV_sink_t_link:
  - Rearrange parameter order: (eigvecs, phase_exp, link_dir, link_max, gauge_link, eigvecs_max, conserved)
  - Remove `t` parameter; gauge_link is now pre-sliced to a specific time by the caller
  - Rename eigvecs_min -> eigvecs_max
  - All parameters except eigvecs now have defaults
  - phase_exp=None defaults to ones((Nx,Nx,Nx,Nc)) (zero momentum)
  - gauge_link=None treated as bool (no gauge link, Case 1)
  - Fix conserved current logic: use gauge_link[3] for temporal direction (previously gauge_link[0])
  - Update conserved case to contract eigvecs_max with eigvecs directly, removing time indexing

4.Increase dynamic_contraction 