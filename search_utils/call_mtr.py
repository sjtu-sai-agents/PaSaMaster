import os
import json
import threading
import os,sys
current_path = os.path.abspath(__file__)
with open(os.path.join(os.path.dirname(current_path), "../config/config.json"), "r") as f:
    config = json.load(f)

class CallMetricsTracker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CallMetricsTracker, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        output_folder = config['metric_path']
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        self.output_folder = output_folder
        self._data_lock = threading.Lock()

        # Initialize metric variables
        self.goo_call = 0
        self.goo_success = 0
        self.goo_time_list = []

        self.keyminder_time_list = []    # Keyminder keyword extraction and query generation time
        self.search_time_list = []       # direct search time
        self.locater_time_list = []      # Locater localization time
        self.ranker_time_list = []       # Ranker scoring time
        self.get_content_time_list = []  # paper content retrieval time

        # org
        self.org_id2ref_ar5iv_call = 0
        self.org_id2ref_ar5iv_success = 0
        self.org_id2ref_ar5iv_time_list = []

        self.org_id2cnt_call = 0
        self.org_id2cnt_success = 0
        self.org_id2cnt_time_list = []

        self.id2paper_call = 0
        self.id2paper_success = 0
        self.id2paper_db_success = 0
        self.id2paper_axv_call = 0
        self.id2paper_axv_success = 0
        self.id2paper_db_time_list = []
        self.id2paper_axv_time_list = []
        self.id2paper_total_time_list = []

        self.org_ti2id_axv_call = 0
        self.org_ti2id_axv_success = 0
        self.org_ti2id_axv_time_list = []

        # new
        self.ti2id_call = 0
        self.ti2id_success = 0
        self.ti2id_db3_success = 0
        self.ti2id_db1_call = 0
        self.ti2id_db1_success = 0
        self.ti2id_axv_call = 0
        self.ti2id_axv_success = 0
        self.ti2id_time_list = []
        self.ti2id_db3_time_list = []
        self.ti2id_db1_time_list = []
        self.ti2id_axv_time_list = []

        self.id2ab_call = 0
        self.id2ab_success = 0
        self.id2ab_db3_success = 0
        self.id2ab_db1_call = 0
        self.id2ab_db1_success = 0
        self.id2ab_axv_call = 0
        self.id2ab_axv_success = 0
        self.id2ab_time_list = []
        self.id2ab_db3_time_list = []
        self.id2ab_db1_time_list = []
        self.id2ab_axv_time_list = []

        self.id2ref_call = 0
        self.id2ref_success = 0
        self.id2ref_ar5iv_call = 0
        self.id2ref_ar5iv_success = 0
        self.id2ref_tex_call = 0
        self.id2ref_tex_success = 0
        self.id2ref_time_list = []
        self.id2ref_ar5iv_time_list = []
        self.id2ref_tex_time_list = []

        self.id2cnt_call = 0
        self.id2cnt_success = 0
        self.id2cnt_ar5iv_call = 0
        self.id2cnt_ar5iv_success = 0
        self.id2cnt_tex_call = 0
        self.id2cnt_tex_success = 0
        self.id2cnt_time_list = []
        self.id2cnt_ar5iv_time_list = []
        self.id2cnt_tex_time_list = []

        self.fetch_eprint_time = []
        self.extract_tar_gz_time = []
        self.parse_tex_folder_time = []
        self.find_tex_time = []
        self.read_tex_time = []
        self.find_bbl_time = []
        self.parse_bbl_time = []
        self.parse_sections_time = []

        self.cnt_fetch_eprint_time = []
        self.cnt_extract_tar_gz_time = []
        self.cnt_parse_tex_folder_time = []
        self.cnt_find_tex_time = []
        self.cnt_read_tex_time = []
        self.cnt_parse_sections_time = []
        
    def add_google(self, success=False, time_cost=None):
        with self._data_lock:
            self.goo_call += 1
            if success:
                self.goo_success += 1
            if time_cost is not None:
                self.goo_time_list.append(time_cost)

    def add_agent_time(self, keyminder=None, search=None, locater=None, ranker=None, get_content=None):
        with self._data_lock:
            if keyminder is not None:
                self.keyminder_time_list.append(keyminder)
            if search is not None:
                self.search_time_list.append(search)
            if locater is not None:
                self.locater_time_list.append(locater)
            if ranker is not None:
                self.ranker_time_list.append(ranker)
            if get_content is not None:
                self.get_content_time_list.append(get_content)

    def org_add_id2ref(self, success=False, time_cost=None):
        with self._data_lock:
            self.org_id2ref_ar5iv_call += 1
            if success:
                self.org_id2ref_ar5iv_success += 1
            if time_cost is not None:
                self.org_id2ref_ar5iv_time_list.append(time_cost)

    def org_add_id2cnt(self, success=False, time_cost=None):
        with self._data_lock:
            self.org_id2cnt_call += 1
            if success:
                self.org_id2cnt_success += 1
            if time_cost is not None:
                self.org_id2cnt_time_list.append(time_cost)

    def add_id2paper(self, db_success=False, axv_call=False, axv_success=False, db_time=None, axv_time=None, total_time=None):
        with self._data_lock:
            self.id2paper_call += 1
            if db_success:
                self.id2paper_db_success += 1
            if axv_call:
                self.id2paper_axv_call += 1
            if axv_success:
                self.id2paper_axv_success += 1
            if db_success or axv_success:
                self.id2paper_success += 1
            if db_time is not None:
                self.id2paper_db_time_list.append(db_time)
            if axv_time is not None:
                self.id2paper_axv_time_list.append(axv_time)
            if total_time is not None:
                self.id2paper_total_time_list.append(total_time)

    def add_ti2id_axv(self, success=False, time_cost=None):
        with self._data_lock:
            self.org_ti2id_axv_call += 1
            if success:
                self.org_ti2id_axv_success += 1
            if time_cost is not None:
                self.org_ti2id_axv_time_list.append(time_cost)

    def add_ti2id(self, success=False, db3_success=False, db1_call=False, db1_success=False, axv_call=False, axv_success=False, time_cost=None, db3_time=None, db1_time=None, axv_time=None):
        with self._data_lock:
            self.ti2id_call += 1
            if success:
                self.ti2id_success += 1
            if db3_success:
                self.ti2id_db3_success += 1
            if db1_call:
                self.ti2id_db1_call += 1
            if db1_success:
                self.ti2id_db1_success += 1
            if axv_call:
                self.ti2id_axv_call += 1
            if axv_success:
                self.ti2id_axv_success += 1
            if time_cost is not None:
                self.ti2id_time_list.append(time_cost)
            if db3_time is not None:
                self.ti2id_db3_time_list.append(db3_time)
            if db1_time is not None:
                self.ti2id_db1_time_list.append(db1_time)
            if axv_time is not None:
                self.ti2id_axv_time_list.append(axv_time)

    def add_id2ab(self, success=False, db3_success=False, db1_call=False, db1_success=False, axv_call=False, axv_success=False, time_cost=None, db3_time=None, db1_time=None, axv_time=None):
        with self._data_lock:
            self.id2ab_call += 1
            if success:
                self.id2ab_success += 1
            if db3_success:
                self.id2ab_db3_success += 1
            if db1_call:
                self.id2ab_db1_call += 1
            if db1_success:
                self.id2ab_db1_success += 1
            if axv_call:
                self.id2ab_axv_call += 1
            if axv_success:
                self.id2ab_axv_success += 1
            if time_cost is not None:
                self.id2ab_time_list.append(time_cost)
            if db3_time is not None:
                self.id2ab_db3_time_list.append(db3_time)
            if db1_time is not None:
                self.id2ab_db1_time_list.append(db1_time)
            if axv_time is not None:
                self.id2ab_axv_time_list.append(axv_time)

    def add_id2ref(self, success=False, tex_success=False, tex_call=False, ar5iv_call=False, ar5iv_success=False, time_cost=None, tex_time=None, ar5iv_time=None):
        with self._data_lock:
            self.id2ref_call += 1
            if success:
                self.id2ref_success += 1
            if tex_success:
                self.id2ref_tex_success += 1
            if tex_call:
                self.id2ref_tex_call += 1
            if ar5iv_call:
                self.id2ref_ar5iv_call += 1
            if ar5iv_success:
                self.id2ref_ar5iv_success += 1
            if time_cost is not None:
                self.id2ref_time_list.append(time_cost)
            if tex_time is not None:
                self.id2ref_tex_time_list.append(tex_time)
            if ar5iv_time is not None:
                self.id2ref_ar5iv_time_list.append(ar5iv_time)

    def add_id2cnt(self, success=False, tex_success=False, tex_call=False, ar5iv_call=False, ar5iv_success=False, time_cost=None, tex_time=None, ar5iv_time=None):
        with self._data_lock:
            self.id2cnt_call += 1
            if success:
                self.id2cnt_success += 1
            if tex_success:
                self.id2cnt_tex_success += 1
            if tex_call:
                self.id2cnt_tex_call += 1
            if ar5iv_call:
                self.id2cnt_ar5iv_call += 1
            if ar5iv_success:
                self.id2cnt_ar5iv_success += 1
            if time_cost is not None:
                self.id2cnt_time_list.append(time_cost)
            if tex_time is not None:
                self.id2cnt_tex_time_list.append(tex_time)
            if ar5iv_time is not None:
                self.id2cnt_ar5iv_time_list.append(ar5iv_time)

    def add_misc_time(self, name, value):
        """
        Generic method that appends value to the specified time array.
        name: string, must be the name of a time array attribute of this class.
        value: numeric value.
        """
        with self._data_lock:
            arr = getattr(self, name, None)
            if arr is not None and isinstance(arr, list):
                arr.append(value)
            else:
                raise AttributeError(f"No such time list: {name}")

    def update_metrics(self, idx, batch_time):
        """
        1. Compute each metric's success rate, average time, and 90/99/100 percentile time, and save to file.
        2. Clear all recorded counts and time arrays.
        """
        import numpy as np
        with self._data_lock:
            def stats(call, success, time_list):
                avg_time = np.mean(time_list) if time_list else 0
                p90 = np.percentile(time_list, 90) if time_list else 0
                p99 = np.percentile(time_list, 99) if time_list else 0
                p100 = np.percentile(time_list, 100) if time_list else 0
                return {
                    'call': call,
                    'success': success,
                    'success_rate': success / call if call else 0,
                    'avg_time': avg_time,
                    'p90_time': p90,
                    'p99_time': p99,
                    'p100_time': p100,
                }

            metrics = {
                'batch_time': batch_time,

                'keyminder': self._time_stats(self.keyminder_time_list),
                'search': self._time_stats(self.search_time_list),
                'locater': self._time_stats(self.locater_time_list),
                'ranker': self._time_stats(self.ranker_time_list),
                'get_content': self._time_stats(self.get_content_time_list),

                'google': stats(self.goo_call, self.goo_success, self.goo_time_list),

                'old': {
                    'id2ref_ar5iv': stats(self.org_id2ref_ar5iv_call, self.org_id2ref_ar5iv_success, self.org_id2ref_ar5iv_time_list),
                    'id2cnt': stats(self.org_id2cnt_call, self.org_id2cnt_success, self.org_id2cnt_time_list),
                    
                    'id2paper_db': stats(self.id2paper_call, self.id2paper_db_success, self.id2paper_db_time_list),
                    'id2paper_axv': stats(self.id2paper_axv_call, self.id2paper_axv_success, self.id2paper_axv_time_list),
                    'id2paper_total': stats(self.id2paper_call, self.id2paper_success, self.id2paper_total_time_list),
                    
                    'ti2id_axv': stats(self.org_ti2id_axv_call, self.org_ti2id_axv_success, self.org_ti2id_axv_time_list),
                },

                'new': {
                    'ti2id': stats(self.ti2id_call, self.ti2id_success, self.ti2id_time_list),
                    'ti2id_db3': stats(self.ti2id_call, self.ti2id_db3_success, self.ti2id_db3_time_list),
                    'ti2id_db1': stats(self.ti2id_db1_call, self.ti2id_db1_success, self.ti2id_db1_time_list),
                    'ti2id_axv': stats(self.ti2id_axv_call, self.ti2id_axv_success, self.ti2id_axv_time_list),

                    'id2ab': stats(self.id2ab_call, self.id2ab_success, self.id2ab_time_list),
                    'id2ab_db3': stats(self.id2ab_call, self.id2ab_db3_success, self.id2ab_db3_time_list),
                    'id2ab_db1': stats(self.id2ab_db1_call, self.id2ab_db1_success, self.id2ab_db1_time_list),
                    'id2ab_axv': stats(self.id2ab_axv_call, self.id2ab_axv_success, self.id2ab_axv_time_list),

                    'id2ref': stats(self.id2ref_call, self.id2ref_success, self.id2ref_time_list),
                    'id2ref_tex': stats(self.id2ref_tex_call, self.id2ref_tex_success, self.id2ref_tex_time_list),
                    'id2ref_ar5iv': stats(self.id2ref_ar5iv_call, self.id2ref_ar5iv_success, self.id2ref_ar5iv_time_list),

                    'id2cnt': stats(self.id2cnt_call, self.id2cnt_success, self.id2cnt_time_list),
                    'id2cnt_tex': stats(self.id2cnt_tex_call, self.id2cnt_tex_success, self.id2cnt_tex_time_list),
                    'id2cnt_ar5iv': stats(self.id2cnt_ar5iv_call, self.id2cnt_ar5iv_success, self.id2cnt_ar5iv_time_list),
                }
            }

            # New: aggregate misc time arrays
            metrics['misc_times'] = {
                'fetch_eprint_time': self._time_stats(self.fetch_eprint_time),
                'extract_tar_gz_time': self._time_stats(self.extract_tar_gz_time),
                'parse_tex_folder_time': self._time_stats(self.parse_tex_folder_time),
                'find_tex_time': self._time_stats(self.find_tex_time),
                'read_tex_time': self._time_stats(self.read_tex_time),
                'find_bbl_time': self._time_stats(self.find_bbl_time),
                'parse_bbl_time': self._time_stats(self.parse_bbl_time),
                'parse_sections_time': self._time_stats(self.parse_sections_time),

                'cnt_fetch_eprint_time': self._time_stats(self.cnt_fetch_eprint_time),
                'cnt_extract_tar_gz_time': self._time_stats(self.cnt_extract_tar_gz_time),
                'cnt_parse_tex_folder_time': self._time_stats(self.cnt_parse_tex_folder_time),
                'cnt_find_tex_time': self._time_stats(self.cnt_find_tex_time),
                'cnt_read_tex_time': self._time_stats(self.cnt_read_tex_time),
                'cnt_parse_sections_time': self._time_stats(self.cnt_parse_sections_time),
            }

            # Save to file
            mtr_file = os.path.join(self.output_folder, f"mtr_{idx}.json")
            with open(mtr_file, "w") as f:
                json.dump(metrics, f, indent=2)

            # Clear all counts and times
            self.goo_call = 0
            self.goo_success = 0
            self.goo_time_list.clear()

            self.org_id2ref_ar5iv_call = 0
            self.org_id2ref_ar5iv_success = 0
            self.org_id2ref_ar5iv_time_list.clear()

            self.org_id2cnt_call = 0
            self.org_id2cnt_success = 0
            self.org_id2cnt_time_list.clear()

            self.id2paper_call = 0
            self.id2paper_success = 0
            self.id2paper_db_success = 0
            self.id2paper_axv_call = 0
            self.id2paper_axv_success = 0
            self.id2paper_db_time_list.clear()
            self.id2paper_axv_time_list.clear()
            self.id2paper_total_time_list.clear()

            self.org_ti2id_axv_call = 0
            self.org_ti2id_axv_success = 0
            self.org_ti2id_axv_time_list.clear()

            self.ti2id_call = 0
            self.ti2id_success = 0
            self.ti2id_db3_success = 0
            self.ti2id_db1_call = 0
            self.ti2id_db1_success = 0
            self.ti2id_axv_call = 0
            self.ti2id_axv_success = 0
            self.ti2id_time_list.clear()
            self.ti2id_db3_time_list.clear()
            self.ti2id_db1_time_list.clear()
            self.ti2id_axv_time_list.clear()

            self.id2ab_call = 0
            self.id2ab_success = 0
            self.id2ab_db3_success = 0
            self.id2ab_db1_call = 0
            self.id2ab_db1_success = 0
            self.id2ab_axv_call = 0
            self.id2ab_axv_success = 0
            self.id2ab_time_list.clear()
            self.id2ab_db3_time_list.clear()
            self.id2ab_db1_time_list.clear()
            self.id2ab_axv_time_list.clear()

            self.id2ref_call = 0
            self.id2ref_success = 0
            self.id2ref_ar5iv_call = 0
            self.id2ref_ar5iv_success = 0
            self.id2ref_tex_call = 0
            self.id2ref_tex_success = 0
            self.id2ref_time_list.clear()
            self.id2ref_tex_time_list.clear()
            self.id2ref_ar5iv_time_list.clear()

            self.id2cnt_call = 0
            self.id2cnt_success = 0
            self.id2cnt_ar5iv_call = 0
            self.id2cnt_ar5iv_success = 0
            self.id2cnt_tex_call = 0
            self.id2cnt_tex_success = 0
            self.id2cnt_time_list.clear()
            self.id2cnt_ar5iv_time_list.clear()
            self.id2cnt_tex_time_list.clear()

            self.keyminder_time_list.clear()
            self.search_time_list.clear()
            self.locater_time_list.clear()
            self.ranker_time_list.clear()
            self.get_content_time_list.clear()

            # Clear misc time arrays
            self.fetch_eprint_time.clear()
            self.extract_tar_gz_time.clear()
            self.parse_tex_folder_time.clear()
            self.find_tex_time.clear()
            self.read_tex_time.clear()
            self.find_bbl_time.clear()
            self.parse_bbl_time.clear()
            self.parse_sections_time.clear()

            self.cnt_fetch_eprint_time.clear()
            self.cnt_extract_tar_gz_time.clear()
            self.cnt_parse_tex_folder_time.clear()
            self.cnt_find_tex_time.clear()
            self.cnt_read_tex_time.clear()
            self.cnt_parse_sections_time.clear()

    def _time_stats(self, time_list):
        import numpy as np
        if not time_list:
            return {'avg_time': 0, 'p90_time': 0, 'p99_time': 0, 'p100_time': 0}
        return {
            'avg_time': float(np.mean(time_list)),
            'p90_time': float(np.percentile(time_list, 90)),
            'p99_time': float(np.percentile(time_list, 99)),
            'p100_time': float(np.percentile(time_list, 100)),
        } 