from deep_kolmogorov.trainer import main, get_args

if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()

    args.gpus = 1
    args.mode = "avg_bs_bermudan_basketput_free_test_num_ex_2_sensor_type_qmc_num_sensor_800"
    main(vars(args))
