#运行代码：                                                                        # 结果输出： result
C_pt_load.py               :读取两点与三点关联函数，得到Ratio--------------------   delta_matirx  (两点，三点关联函数 )  Ratio_data   (Ratio 数据 )   

fit_ratio.py               :拟合ratio 得到 c0 (动量空间的准PDF)------------------   ratio_fit_result (c0 关于 z 的分布) ratio_tsep_tisep (所有参数的详细分布)

hB_data.py                 :插值得到 用于下一步 重整因子Z 拟合的c0 数据 ----------  hB_data

fit_zr.py                  :拟合Pz = 0 的c0 得到重整因子Z------------------------   ZR_fit_result  (重整化因子Z的拟合结果)

fit_hR_big_lambda.py       :对重整化后的准PDF作lambda外推并作傅里叶变换---------------    hR_x            (准PDF)

matching.py                :将准PDF 转化为光锥PDF--------------------------------   hR_PDF          (光锥PDF)

fit_pz_a_extrapolatiing.py :将得到的光锥PDF做格距和动量外推------------------------   hR_res


fit_2pt.py                 : 读取 2pt 数据， 拟合得到E0 --------------------------   par_2pt 

fit_E0.py                  : 读取par_2pt 中拟合得到的E0数据，拟合得到 m, k2--------   par_E0



# 辅助代码：
data_chose.py               : 用于选取拟合的Ratio 数据