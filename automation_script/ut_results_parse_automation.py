#!/usr/bin/env python3

"""This script parses PyTorch unit tests results based on configurations provided.
The output will be written to automation_logs folder

"""

import argparse
import subprocess
import os
import shutil

from datetime import datetime
from pathlib import Path
from parse_xml_results import (
        parse_xml_report
)

# unit test status list
UT_STATUS_LIST = [
    "PASSED",
    "MISSED",
    "SKIPPED",
    "FAILED",
    "XFAILED",
    "ERROR"
]

DEFAULT_CORE_TESTS = [
    "test_nn",
    "test_torch",
    "test_cuda",
    "test_ops",
    "test_unary_ufuncs",
    "test_autograd",
    "inductor/test_torchinductor"
]

DISTRIBUTED_CORE_TESTS = [
    "distributed/test_c10d_common",
    "distributed/test_c10d_nccl",
    "distributed/test_distributed_spawn"
]

def parse_xml_reports_as_dict(workflow_run_id, workflow_run_attempt, tag, workflow_name, path="."):
    test_cases = {}
    items_list = os.listdir(path)
    for dir in items_list:
        new_dir = path + '/' + dir + '/'
        if os.path.isdir(new_dir):
            for xml_report in Path(new_dir).glob("**/*.xml"):
                test_cases.update(
                    parse_xml_report(
                        tag,
                        xml_report,
                        workflow_run_id,
                        workflow_run_attempt,
                        workflow_name
                    )
                )
    return test_cases

def get_test_status(test_case):
  # In order of priority: S=skipped, F=failure, E=error, P=pass
  if "skipped" in test_case and test_case["skipped"]:
      type_message = test_case["skipped"]
      if type_message.__contains__('type') and type_message['type'] == "pytest.xfail":
          return "XFAILED"
      else:
          return "SKIPPED"
  elif "failure" in test_case and test_case["failure"]:
    return "FAILED"
  elif "error" in test_case and test_case["error"]:
    return "ERROR"
  else:
    return "PASSED"

def get_test_message(test_case, status=None):
  if status == "SKIPPED":
    return test_case["skipped"] if "skipped" in test_case else ""
  elif status == "FAILED":
    return test_case["failure"] if "failure" in test_case else ""
  elif status == "ERROR":
    return test_case["error"] if "error" in test_case else ""
  else:
    if "skipped" in test_case:
      return test_case["skipped"]
    elif "failure" in test_case:
      return test_case["failure"]
    elif "error" in test_case:
      return test_case["error"]
    else:
      return ""

def get_test_running_time(test_case):
  if test_case.__contains__('time'):
    return test_case["time"]
  return ""

def summarize_xml_files(path, workflow_name):
    # results: list of dict
    pass_dict = {}
    skip_dict = {}
    xfail_dict = {}
    fail_dict = {}
    error_dict = {}
    statistics_dict = {}
    # statistics
    TOTAL_TEST_NUM = 0
    TOTAL_PASSED_NUM = 0
    TOTAL_SKIPPED_NUM = 0
    TOTAL_XFAIL_NUM = 0
    TOTAL_FAILED_NUM = 0
    TOTAL_ERROR_NUM = 0

    #parse the xml files
    test_cases = parse_xml_reports_as_dict(-1, -1, 'testcase', workflow_name, path)
    test_cases_summary_csv = {}
    for (k,v) in list(test_cases.items()):
        item_values = {}
        item_values["test_file"] = k[0]
        item_values["test_class"] = k[1]
        item_values["test_name"] = k[2]
        test_status = get_test_status(v)
        item_values["status"] = test_status
        test_running_time = get_test_running_time(v)
        item_values["running_time"] = test_running_time
        test_message = get_test_message(v, test_status)
        item_values["message"] = test_message
        TOTAL_TEST_NUM += 1
        item_info_key = k[0] + "::" + k[1] + "::" + k[2]
        item_info_value = ""
        if test_status == "PASSED":
            item_info_value = str(test_running_time)
            TOTAL_PASSED_NUM += 1
            pass_dict[item_info_key] = item_info_value
        elif test_status == "SKIPPED":
            item_info_value = str(test_running_time)
            TOTAL_SKIPPED_NUM += 1
            skip_dict[item_info_key] = item_info_value
        elif test_status == "XFAILED":
            item_info_value = str(test_running_time)
            TOTAL_XFAIL_NUM += 1
            xfail_dict[item_info_key] = item_info_value
        elif test_status == "FAILED":
            item_info_value = test_message
            TOTAL_FAILED_NUM += 1
            fail_dict[item_info_key] = item_info_value
        elif test_status == "ERROR":
            item_info_value = test_message
            TOTAL_ERROR_NUM += 1
            error_dict[item_info_key] = item_info_value
        test_cases_summary_csv[k] = item_values

    # generate statistics_dict
    statistics_dict["TOTAL"] = TOTAL_TEST_NUM
    statistics_dict["PASSED"] = TOTAL_PASSED_NUM
    statistics_dict["SKIPPED"] = TOTAL_SKIPPED_NUM
    statistics_dict["XFAILED"] = TOTAL_XFAIL_NUM
    statistics_dict["FAILED"] = TOTAL_FAILED_NUM
    statistics_dict["ERROR"] = TOTAL_ERROR_NUM

    # generate results
    res = {}
    res["pass"] = pass_dict
    res["skip"] = skip_dict
    res["xfail"] = xfail_dict
    res["fail"] = fail_dict
    res["error"] = error_dict
    res["statistics"] = statistics_dict

    return res

def run_entire_tests(workflow_name, overall_logs_path_current_run, test_reports_src):
    # test_reports_src = '/var/lib/jenkins/pytorch/test/test-reports/'
    # overall_logs_path_current_run = "/var/lib/jenkins/pytorch/automation_logs/" + str_current_datetime + "/"
    if os.path.exists(test_reports_src):
        shutil.rmtree(test_reports_src)

    os.mkdir(test_reports_src)
    copied_logs_path = ""
    if workflow_name == "default":
        os.environ['TEST_CONFIG'] = 'default'
        copied_logs_path = overall_logs_path_current_run + "default_xml_results_entire_tests/"
    elif workflow_name == "distributed":
        os.environ['TEST_CONFIG'] = 'distributed'
        copied_logs_path = overall_logs_path_current_run + "distributed_xml_results_entire_tests/"
    elif workflow_name == "inductor":
        os.environ['TEST_CONFIG'] = 'inductor'
        copied_logs_path = overall_logs_path_current_run + "inductor_xml_results_entire_tests/"
    # use test.sh for tests execution
    command = "/var/lib/jenkins/pytorch/.ci/pytorch/test.sh"
    subprocess.call(command, shell=True)
    copied_logs_path_destination = shutil.copytree(test_reports_src, copied_logs_path)
    entire_results_dict = summarize_xml_files(copied_logs_path_destination, workflow_name)
    
    return entire_results_dict

def run_priority_tests(workflow_name, overall_logs_path_current_run, test_reports_src):
    # test_reports_src = '/var/lib/jenkins/pytorch/test/test-reports/'
    # overall_logs_path_current_run = "/var/lib/jenkins/pytorch/automation_logs/" + str_current_datetime + "/"
    if os.path.exists(test_reports_src):
        shutil.rmtree(test_reports_src)

    os.mkdir(test_reports_src)
    copied_logs_path = ""
    if workflow_name == "default":
        os.environ['TEST_CONFIG'] = 'default'
        os.environ['HIP_VISIBLE_DEVICES'] = '0'
        copied_logs_path = overall_logs_path_current_run + "default_xml_results_priority_tests/"
        # use run_test.py for tests execution
        default_priority_test_suites = " ".join(DEFAULT_CORE_TESTS)
        command = "python /var/lib/jenkins/pytorch/test/run_test.py --include " + default_priority_test_suites + " --continue-through-error --exclude-jit-executor --exclude-distributed-tests --verbose"
        subprocess.call(command, shell=True)
    elif workflow_name == "distributed":
        os.environ['TEST_CONFIG'] = 'distributed'
        os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
        copied_logs_path = overall_logs_path_current_run + "distributed_xml_results_priority_tests/"
        # use run_test.py for tests execution
        distributed_priority_test_suites = " ".join(DISTRIBUTED_CORE_TESTS)
        command = "python /var/lib/jenkins/pytorch/test/run_test.py --include " + distributed_priority_test_suites + " --continue-through-error --distributed-tests --verbose"
        subprocess.call(command, shell=True)
    copied_logs_path_destination = shutil.copytree(test_reports_src, copied_logs_path)
    priority_results_dict = summarize_xml_files(copied_logs_path_destination, workflow_name)

    return priority_results_dict

def run_selected_tests(workflow_name, overall_logs_path_current_run, test_reports_src, selected_list):
    # test_reports_src = '/var/lib/jenkins/pytorch/test/test-reports/'
    # overall_logs_path_current_run = "/var/lib/jenkins/pytorch/automation_logs/" + str_current_datetime + "/"
    if os.path.exists(test_reports_src):
        shutil.rmtree(test_reports_src)

    os.mkdir(test_reports_src)
    copied_logs_path = ""
    if workflow_name == "default":
        os.environ['TEST_CONFIG'] = 'default'
        os.environ['HIP_VISIBLE_DEVICES'] = '0'
        copied_logs_path = overall_logs_path_current_run + "default_xml_results_selected_tests/"
        # use run_test.py for tests execution
        default_selected_test_suites = " ".join(selected_list)
        command = "python /var/lib/jenkins/pytorch/test/run_test.py --include " + default_selected_test_suites  + " --continue-through-error --exclude-jit-executor --exclude-distributed-tests --verbose"
        subprocess.call(command, shell=True)
    elif workflow_name == "distributed":
        os.environ['TEST_CONFIG'] = 'distributed'
        os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
        copied_logs_path = overall_logs_path_current_run + "distributed_xml_results_selected_tests/"
        # use run_test.py for tests execution
        distributed_selected_test_suites = " ".join(selected_list)
        command = "python /var/lib/jenkins/pytorch/test/run_test.py --include " + distributed_selected_test_suites + " --continue-through-error --distributed-tests --verbose"
        subprocess.call(command, shell=True)
    elif workflow_name == "inductor":
        os.environ['TEST_CONFIG'] = 'inductor'
        os.environ['HIP_VISIBLE_DEVICES'] = '0'
        copied_logs_path = overall_logs_path_current_run + "inductor_xml_results_selected_tests/"
        inductor_selected_test_suites = ""
        non_inductor_selected_test_suites = ""
        for item in selected_list:
            if "inductor/" in item:
                inductor_selected_test_suites += item
                inductor_selected_test_suites += " "
            else:
                non_inductor_selected_test_suites += item
                non_inductor_selected_test_suites += " "
        if inductor_selected_test_suites != "":
            inductor_selected_test_suites = inductor_selected_test_suites[:-1]
            command = "python /var/lib/jenkins/pytorch/test/run_test.py --include " + non_inductor_selected_test_suites + " --continue-through-error --verbose"
            subprocess.call(command, shell=True)
        if non_inductor_selected_test_suites != "":
            non_inductor_selected_test_suites = non_inductor_selected_test_suites[:-1]
            command = "python /var/lib/jenkins/pytorch/test/run_test.py --inductor --include " + non_inductor_selected_test_suites + " --continue-through-error --verbose"
            subprocess.call(command, shell=True)
    copied_logs_path_destination = shutil.copytree(test_reports_src, copied_logs_path)
    selected_results_dict = summarize_xml_files(copied_logs_path_destination, workflow_name)

    return selected_results_dict

def run_test_and_summarize_results(args):
    # copy current environment variables
    _environ = dict(os.environ)

    # all test results dict
    res_all_tests_dict = {}

    # create logs folder
    repo_test_log_folder_path = "/var/lib/jenkins/pytorch/automation_logs/"
    if not os.path.exists(repo_test_log_folder_path):
        os.mkdir(repo_test_log_folder_path)

    # Set common environment variables for all scenarios
    os.environ['CI'] = '1'
    os.environ['PYTORCH_TEST_WITH_ROCM'] = '1'
    os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
    os.environ['PYTORCH_TESTING_DEVICE_ONLY_FOR'] = 'cuda'
    os.environ['CONTINUE_THROUGH_ERROR'] = 'True'

    test_reports_src = '/var/lib/jenkins/pytorch/test/test-reports/'

    # Time stamp
    current_datetime = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    print("Current date & time : ", current_datetime)
    # performed as Job ID
    str_current_datetime = str(current_datetime)
    overall_logs_path_current_run = repo_test_log_folder_path + str_current_datetime + "/"
    # Run entire tests for each workflow
    if not args.priority_test and not args.default_list and not args.distributed_list and not args.inductor_list:
        # run entire tests for default, distributed and inductor workflows → use test.sh
        if not args.test_config:
            # default test process
            res_default_all = run_entire_tests("default", overall_logs_path_current_run, test_reports_src)
            res_all_tests_dict["default_all"] = res_default_all
            # distributed test process
            res_distributed_all = run_entire_tests("distributed", overall_logs_path_current_run, test_reports_src)
            res_all_tests_dict["distributed_all"] = res_distributed_all
            # inductor test process
            res_inductor_all = run_entire_tests("inductor", overall_logs_path_current_run, test_reports_src)
            res_all_tests_dict["inductor_all"] = res_inductor_all
        else:
            workflow_list = []
            for item in args.test_config:
                workflow_list.append(item)
            if "default" in workflow_list:
                res_default_all = run_entire_tests("default", overall_logs_path_current_run, test_reports_src)
                res_all_tests_dict["default_all"] = res_default_all
            if "distributed" in workflow_list:
                res_distributed_all = run_entire_tests("distributed", overall_logs_path_current_run, test_reports_src)
                res_all_tests_dict["distributed_all"] = res_distributed_all
            if "inductor" in workflow_list:
                res_inductor_all = run_entire_tests("inductor", overall_logs_path_current_run, test_reports_src)
                res_all_tests_dict["inductor_all"] = res_inductor_all
    # Run priority test for each workflow
    elif args.priority_test and not args.default_list and not args.distributed_list and not args.inductor_list:
        if not args.test_config:
            # default test process
            res_default_priority = run_priority_tests("default", overall_logs_path_current_run, test_reports_src)
            res_all_tests_dict["default_priority"] = res_default_priority
            # distributed test process
            res_distributed_priority = run_priority_tests("distributed", overall_logs_path_current_run, test_reports_src)
            res_all_tests_dict["distributed_priority"] = res_distributed_priority
            # will not run inductor priority tests
            print("Will not run inductor priority tests since they are not defined!")
        else:
            workflow_list = []
            for item in args.test_config:
                workflow_list.append(item)
            if "default" in workflow_list:
                res_default_priority = run_priority_tests("default", overall_logs_path_current_run, test_reports_src)
                res_all_tests_dict["default_priority"] = res_default_priority
            if "distributed" in workflow_list:
                res_distributed_priority = run_priority_tests("distributed", overall_logs_path_current_run, test_reports_src)
                res_all_tests_dict["distributed_priority"] = res_distributed_priority
            if "inductor" in workflow_list:
                print("Will not run inductor priority tests since they are not defined!")
    # Run specified tests for each workflow
    elif (args.default_list or args.distributed_list or args.inductor_list) and not args.test_config and not args.priority_test:
        if args.default_list:
            default_workflow_list = []
            for item in args.default_list:
                default_workflow_list.append(item)
            res_default_selected = run_selected_tests("default", overall_logs_path_current_run, test_reports_src, default_workflow_list)
            res_all_tests_dict["default_selected"] = res_default_selected
        if args.distributed_list:
            distributed_workflow_list = []
            for item in args.distributed_list:
                distributed_workflow_list.append(item)
            res_distributed_selected = run_selected_tests("distributed", overall_logs_path_current_run, test_reports_src, distributed_workflow_list)
            res_all_tests_dict["distributed_selected"] = res_distributed_selected
        if args.inductor_list:
            inductor_workflow_list = []
            for item in args.inductor_list:
                 inductor_workflow_list.append(item)
            res_inductor_selected = run_selected_tests("inductor", overall_logs_path_current_run, test_reports_src, inductor_workflow_list)
            res_all_tests_dict["inductor_selected"] = res_inductor_selected
    else:
        raise Exception("Invalid test configurations!")

    # restore environment variables
    os.environ.clear()
    os.environ.update(_environ)

    return res_all_tests_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Run PyTorch unit tests and generate xml results summary')
    parser.add_argument('--test_config', nargs='+', default=[], type=str, help="test workflows to be executed")
    parser.add_argument('--priority_test', action='store_true', help="run priority tests only")
    parser.add_argument('--default_list', nargs='+', default=[], help="default tests to be executed")
    parser.add_argument('--distributed_list', nargs='+', default=[], help="distributed tests to be executed")
    parser.add_argument('--inductor_list', nargs='+', default=[], help="inductor tests to be executed")
    return parser.parse_args()

def main():
    global args
    args = parse_args()
    all_tests_results = {}
    all_tests_results = run_test_and_summarize_results(args)

if __name__ == "__main__":
    main()