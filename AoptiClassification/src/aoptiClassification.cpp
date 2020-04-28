#include "aoptiClassification.h"



AoptiClassification::AoptiClassification(void)
{
}

bool AoptiClassification::InitClassification(std::string pretrainedModelPath, std::string pretrainedModelName,int num_classes)
{
	this->lib_path = pretrainedModelPath + '/' + pretrainedModelName + ".dll";
	this->json_path = pretrainedModelPath + '/' + pretrainedModelName + ".json";
	this->param_path = pretrainedModelPath + '/' + pretrainedModelName + ".params";
	this->num_classes = num_classes;

	//�ж������ļ��Ƿ����
	std::ifstream lib_in(lib_path);
	std::ifstream json_in(json_path, std::ios::in);
	std::ifstream params_in(param_path, std::ios::binary);

	if (!lib_in.is_open()) {
		error_message = "dll�ļ���ȡ��������" + lib_path + "·���Ƿ���ȷ��";
		return false;
	}
	if (!json_in.is_open()) {
		error_message = "json�ļ���ȡ��������" + json_path + "·���Ƿ���ȷ��";
		return false;
	}
	if (!params_in.is_open()) {
		error_message = "params�ļ���ȡ��������" + json_path + "·���Ƿ���ȷ��";
		return false;
	}

	//ģ���ļ����أ���ʼ��
	// tvm module for compiled functions
	tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(lib_path);

	// json graph
	std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
	json_in.close();

	// parameters in binary
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

	int in_ndim = 4;
	int64_t in_shape[4] = { 1, 3, 300, 300 };
	TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);

	// get the function from the module(set input data)
	set_input = mod.GetFunction("set_input");

	// get the function from the module(load patameters)
	tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
	load_params(params_arr);

	int out_ndim = 2;
	int64_t out_shape[2] = { 1, num_classes};
	TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

	// get the function from the module(run it)
	run = mod.GetFunction("run");
	// get the function from the module(get output data)
	get_output = mod.GetFunction("get_output");


	return true;
}

/*
ͼƬԤ����resize+norm
*/
void AoptiClassification::ProcessData(cv::Mat &img)
{
	//resize��norm��һ��
	cv::Mat resize_img;
	cv::resize(img, img, cv::Size(300, 300), 0, 0, cv::INTER_LINEAR);
	img.convertTo(img, CV_32F, 1.0 / 255, 0);

	//vconcat
	std::vector<cv::Mat> matrices = { img,img,img };
	cv::vconcat(matrices, img);
}

const std::string AoptiClassification::GetErrorMessage()
{
	return error_message;
}

int AoptiClassification::Classification(uchar *pImg, int nHeight, int nWidth)
{
	if (pImg == nullptr)
	{
		return false;
	}
	int nByte = nHeight * nWidth;//�ֽڼ���
	int nType = CV_8UC1;
	cv::Mat outImg = cv::Mat::zeros(nHeight, nWidth, nType);
	memcpy(outImg.data, pImg, nByte);

	//ͼƬԤ����
	ProcessData(outImg);
	// set img.data to x->data
	x->data = outImg.data;
	set_input("input", x);
	run();
	get_output(0, y);

	// get the maximum position in output vector
	float* y_iter = static_cast<float*>(y->data);

	auto max_iter = std::max_element(y_iter, y_iter + num_classes);
	auto max_index = std::distance(y_iter, max_iter);

	return max_index;
}

AoptiClassification::~AoptiClassification()
{
	if (y != NULL) {
		TVMArrayFree(y);
	}
}
