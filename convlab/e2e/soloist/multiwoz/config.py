import logging, time, os

class _Config:
    def __init__(self):
        self._multiwoz_soloist_init()

    def _multiwoz_soloist_init(self):
        self.dbs = {
            'attraction': 'db/attraction_db_processed.json',
            'hospital': 'db/hospital_db_processed.json',
            'hotel': 'db/hotel_db_processed.json',
            'police': 'db/police_db_processed.json',
            'restaurant': 'db/restaurant_db_processed.json',
            'taxi': 'db/taxi_db_processed.json',
            'train': 'db/train_db_processed.json',
        }


        self.prefix =  ""
        self.max_length = 512
        self.padding = "max_length"
        self.surfix = '=>'
        self.max_target_length = 196
        self.model_name_or_path = 'soloist-model'
        self.num_beams = 5
        self.ignore_pad_token_for_loss = True

        self.dataset = 'multiwoz_dataset.py'
        self.dataset_name = 'multiwoz_dataset.py'
        self.train_file = None
        self.validation_file = None
        self.max_source_length = None
        self.source_prefix = ""
        self.preprocessing_num_workers = 1
        self.max_target_length = 192
        self.val_max_target_length = 192
        self.max_length = 512
        self.num_beams = 5
        self.config_name = None
        self.tokenizer_name = None
        self.use_slow_tokenizer = False
        self.per_device_train_batch_size = 6
        self.per_device_eval_batch_size = 6
        self.learning_rate = 5e-5
        self.weight_decay = 0
        self.num_train_epochs = 10
        self.max_train_steps = None
        self.gradient_accumulation_steps = 1
        self.lr_scheduler_type = "linear"
        self.num_warmup_steps = 0
        self.output_dir = './temp'
        self.seed = 2021
        self.model_type = None
        self.overwrite_cache = False
        self.pad_to_max_length = True
        self.ignore_pad_token_for_loss = True
        self.logging_steps = 1000
        self.save_steps = 10000
        self.save_every_checkpoint = True
        self.max_grad_norm = 1.0
        self.no_kb = False
        self.use_special_token = False
        self.format_version = 'e2e'
        self.wandb_exp_name  = 'mwoz'

        self.top_p = 0.9

        self.cuda = True


    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s


    def _init_logging_handler(self, mode):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists('./log'):
            os.mkdir('./log')
        if self.save_log and self.mode == 'train':
            file_handler = logging.FileHandler('./log/log_{}_{}_{}_{}_sd{}.txt'.format(self.log_time, mode, '-'.join(self.exp_domains), self.exp_no, self.seed))
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        elif self.mode == 'test':
            eval_log_path = os.path.join(self.eval_load_path, 'eval_log.json')
            # if os.path.exists(eval_log_path):
            #     os.remove(eval_log_path)
            file_handler = logging.FileHandler(eval_log_path)
            logging.basicConfig(handlers=[stderr_handler, file_handler])
        else:
            logging.basicConfig(handlers=[stderr_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()

