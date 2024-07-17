from deep_kolmogorov.trainer import main, get_args

if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()

    args.gpus = 1
    # args.mode = "avg_bs_bermudan_put_free_test_num_ex_2_qmax_0.1_sensor_type_MP_num_sensor_200"
    # main(vars(args))
    # args.mode = "avg_bs_bermudan_put_free_test_num_ex_2_qmax_0.1_sensor_type_MP_num_sensor_200_var_rescale_2"
    # main(vars(args))

    # args.mode = "avg_bs_bermudan_put_free_test_num_ex_4_qmax_0.1_sensor_type_MP_num_sensor_200"
    # main(vars(args))
    args.mode = "avg_bs_bermudan_put_free_test_num_ex_4_qmax_0.1_sensor_type_MP_num_sensor_50_var_rescale_2"
    main(vars(args))