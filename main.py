from deep_kolmogorov.trainer import main, get_args

if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()

    args.mode = 'avg_bs_bermudan_put_free_test_num_ex_2_max_4.4_var_scale_1_length_scale_100'
    args.gpus = 1
    main(vars(args))