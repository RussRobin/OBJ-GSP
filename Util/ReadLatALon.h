#pragma once

// CWX NOTE this file shouldn't be included in the .sln project

#include <gdal.h>
#include <gdalexif.h>
#include <gdal_priv.h>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <iomanip>
#include <fstream>
#define ACCEPT_USE_OF_DEPRECATED_PROJ_API_H 1
#include "ogrsf_frmts.h"
#include "ogr_srs_api.h"
#include "ogr_spatialref.h"
#include "ogr_api.h"
#include "proj_api.h"
#include "../Configure.h"
#include "./Transform.h"
#define BYTE short 

using namespace std;

vector <pair<double, double>> ReadJPGLonALon(vector<string> ImageName);

pair<double, double> ReadJPGLonALon(string ImageName); //单个图像

vector <pair<double, double>> ReadTIFLonALon(vector<string> ImageName);

pair<double, double> ReadTIFLonALon(string ImageName); //单个图像

/* 大疆精灵4 多光谱版 需要从xmp元数据中读取*/
vector <pair<double, double>> ReadPhantom4TIFLonALon(vector<string> ImageName);

pair<double, double> ReadPhantom4TIFLonALon(string ImageName); //单个图像

map<string, pair<double, double>> ReadLatALonByGPS(string GPS_Path);

map<string, pair<double, double>> ReadLatALonByTXT(string GPS_Path);
