from deep_kolmogorov.trainer import main, get_args

if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()

    args.mode = 'avg_bs_bermudan_put_free_test_num_ex_12_qmax_0.05_sensor_type_MP_num_sensor_50'
    args.gpus = 1
    main(vars(args))