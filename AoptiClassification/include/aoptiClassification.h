#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

//������Ϊdll
#ifndef DLL_API
#define DLL_API __declspec(dllexport)
#endif

class DLL_API AoptiClassification
{
private:
	//ģ��dll�ļ�·��
	std::string lib_path;

	//����ṹjson�ļ�·��
	std::string json_path;

	//��������ļ�·��
	std::string param_path;

	//��������
	int num_classes;

	//������Ϣ
	std::string error_message;

	//ģ�ͺ����Ͷ�ȡ����
	tvm::runtime::PackedFunc set_input;
	tvm::runtime::PackedFunc run;
	tvm::runtime::PackedFunc get_output;

	//��������
	DLTensor* x;
	DLTensor* y;

	//ͼƬԤ����
	virtual void ProcessData(cv::Mat &img);

public:
	AoptiClassification(void);
	
	//��ʼ��ģ��
	bool InitClassification(std::string pretrainedModelPath, std::string pretrainedModelName, int num_classes);

	//��ȡ������Ϣ
	const std::string GetErrorMessage();

	//��ȡ������
	int Classification(uchar *pImg, int nHeight, int nWidth);

	//�����������ͷ��ڴ�
	~AoptiClassification();
};

