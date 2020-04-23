#include<iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <algorithm>
#include <fstream>
#include <ctime>
#include <iterator>
#include <stdexcept>
#include <string>
#include <ctime>

/*
图像预处理：
1.resize 双线性插值
2.vconcat 填充三通道
*/
void ProcessData(cv::Mat &img) {
	//resize，norm归一化
	cv::Mat resize_img;
	cv::resize(img, img, cv::Size(300, 300), 0, 0, cv::INTER_LINEAR);
	img.convertTo(img, CV_32F, 1.0 / 255, 0);

	//vconcat
	std::vector<cv::Mat> matrices = { img,img,img };
	cv::vconcat(matrices, img);
	//std::cout << "拼接之后图片大小：" << img << std::endl;

}
/*
使用tvm_runtime推理
*/
void TvmInference(std::string data_path, std::string model_name) {

	// tvm module for compiled functions
	tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(data_path + "/lib/" + model_name + ".dll");

	// json graph
	std::ifstream json_in(data_path + "/lib/" + model_name + ".json", std::ios::in);
	std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
	json_in.close();

	// parameters in binary
	std::ifstream params_in(data_path + "/lib/" + model_name + ".params", std::ios::binary);
	std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
	params_in.close();

	// parameters need to be TVMByteArray type to indicate the binary data
	TVMByteArray params_arr;
	params_arr.data = params_data.c_str();
	params_arr.size = params_data.length();

	int dtype_code = kDLFloat;
	int dtype_bits = 32;
	int dtype_lanes = 1;
	int device_type = kDLCPU;
	int device_id = 0;

	// get global function module for graph runtime
	tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);

	DLTensor* x;
	int in_ndim = 4;
	int64_t in_shape[4] = { 1, 3, 300, 300 };
	TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);

	// get the function from the module(set input data)
	tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");

	// get the function from the module(load patameters)
	tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
	load_params(params_arr);

	DLTensor* y;
	int out_ndim = 2;
	int64_t out_shape[2] = { 1, 7, };
	TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

	// get the function from the module(run it)
	tvm::runtime::PackedFunc run = mod.GetFunction("run");
	// get the function from the module(get output data)
	tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");

	//1.class2id字典
	std::map<std::string, int> class2id = { {"ddust",0} ,{"filament",1},{"glue",2},{"good",3},{"scratch",4},{"silk",5},{"smudge",6} };
	std::map<std::string, int>::iterator iter;
	//iter = class2id.begin();

	float total_num = 0;
	float each_class_num = 0;
	int total_cor_num = 0;
	int each_class_cor_num = 0;
	int total_time = 0;
	time_t start, end;
	cv::Mat img;


	//2.获取根目录下所有文件夹
	for (iter = class2id.begin(); iter != class2id.end(); iter++)
	{
		//3依次获取文件夹所有图片进行tvm推理
		std::string pattern = data_path + "/test/" + iter->first + "/*.tif";
		std::vector<cv::String> file_names;
		cv::glob(pattern, file_names);
		std::cout << iter->first << "类别共" << file_names.size() << "张图片，开始推理..." << std::endl;

		each_class_num = file_names.size();
		for (size_t i = 0; i < each_class_num; i++)
		{
			img = cv::imread(file_names[i], 0);
			//预处理
			start = time(0);
			ProcessData(img);
			// set img.data to x->data
			x->data = img.data;
			set_input("input", x);
			run();
			get_output(0, y);

			// get the maximum position in output vector
			float* y_iter = static_cast<float*>(y->data);
			/*	std::cout << *y_iter << "\t" << *(y_iter + 1) << "\t" << *(y_iter + 2) << "\t" << *(y_iter + 3)
					<< "\t" << *(y_iter + 4) << "\t" << *(y_iter + 5) << "\t" << *(y_iter + 6) << std::endl;*/
			auto max_iter = std::max_element(y_iter, y_iter + 7);
			auto max_index = std::distance(y_iter, max_iter);
			end = time(0);

			//累加推理正确的图片
			if (max_index == iter->second) {
				each_class_cor_num += 1;
				total_cor_num += 1;
			}

			//时间和推理图片累加
			total_time += (end - start);
			total_num += 1;
		}

		std::cout << "预测正确的图片有：" << each_class_cor_num << "，准确率：" << each_class_cor_num / each_class_num << std::endl;
		//当前类别正确数量清0
		each_class_cor_num = 0;
	}

	std::cout << "inference times" << total_num << "\tcost time" << total_time << "s" << std::endl;
	std::cout << "average time:\t" << (total_time / total_num) * 1000 << "ms/img" << std::endl;
	std::cout << "accuracy:\t" << total_cor_num / total_num << std::endl;


	//释放对象内存
	TVMArrayFree(x);
	TVMArrayFree(y);

}

int main() {
	std::string data_path = "D:/VSProjects/data";
	std::string model_name = "squeezenet1-1-9791llvm -mcpu=core-avx2";
	//std::string data_path = "./data";
	TvmInference(data_path, model_name);
	return 0;
}