#include <gdal_priv.h>
#include <ogr_spatialref.h>
#include <opencv2/core.hpp>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("resource_info raster output_json");
 GDALAllRegister();std::unique_ptr<GDALDataset,void(*)(GDALDataset*)> ds(static_cast<GDALDataset*>(GDALOpen(argv[1],GA_ReadOnly)),[](GDALDataset*p){if(p)GDALClose(p);});
 if(!ds)throw std::runtime_error("cannot open raster");double gt[6];if(ds->GetGeoTransform(gt)!=CE_None)throw std::runtime_error("missing affine");
 int w=ds->GetRasterXSize(),h=ds->GetRasterYSize();auto*b=ds->GetRasterBand(1);float center;
 if(b->RasterIO(GF_Read,w/2,h/2,1,1,&center,1,1,GDT_Float32,0,0)!=CE_None)throw std::runtime_error("center read failed");
 int hn,hs,ho;double nodata=b->GetNoDataValue(&hn),scale=b->GetScale(&hs),offset=b->GetOffset(&ho);if(!hs)scale=1;if(!ho)offset=0;
 bool center_valid=std::isfinite(center)&&(!hn||center!=float(nodata));
 OGRSpatialReference source;source.SetFromUserInput(ds->GetProjectionRef());source.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
 std::unique_ptr<OGRSpatialReference> geographic(source.CloneGeogCS());if(!geographic)throw std::runtime_error("missing geographic CRS");geographic->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
 std::unique_ptr<OGRCoordinateTransformation,void(*)(OGRCoordinateTransformation*)> projection(OGRCreateCoordinateTransformation(&source,geographic.get()),[](OGRCoordinateTransformation*p){if(p)OCTDestroyCoordinateTransformation(p);});
 if(!projection)throw std::runtime_error("cannot convert projected coordinates to lunar longitude/latitude");
 double origin_x=gt[0]+(w/2+.5)*gt[1]+(h/2+.5)*gt[2],origin_y=gt[3]+(w/2+.5)*gt[4]+(h/2+.5)*gt[5];
 double lon=origin_x,lat=origin_y;if(!projection->Transform(1,&lon,&lat))throw std::runtime_error("center geographic transform failed");
 char*geowkt=nullptr;geographic->exportToWkt(&geowkt);
 cv::FileStorage out(argv[2],cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);if(!out.isOpened())throw std::runtime_error("cannot open metadata output");
 out<<"source_path"<<std::filesystem::absolute(argv[1]).string()<<"width"<<w<<"height"<<h<<"bands"<<ds->GetRasterCount();
 out<<"crs_wkt"<<ds->GetProjectionRef()<<"geographic_crs_wkt"<<geowkt;CPLFree(geowkt);
 out<<"geotransform"<<"[";for(double value:gt)out<<value;out<<"]";
 out<<"scale"<<scale<<"offset"<<offset<<"has_nodata"<<hn;if(hn)out<<"nodata"<<nodata;
 out<<"origin_projected_m"<<"["<<origin_x<<origin_y<<"]"<<"origin_altitude_source_m"<<(center_valid?center*scale+offset:0.)<<"origin_valid"<<int(center_valid);
 out<<"origin_lon_lat_deg"<<"["<<lon<<lat<<"]"<<"reference_sphere_radius_m"<<geographic->GetSemiMajor();
 out<<"coordinate_frame"<<"local_projected_raster_x_y_and_source_height_relative_to_center";
 out<<"vertical_datum"<<"as_provided_in_source_raster_not_independently_verified";
 out<<"corners_lon_lat_deg"<<"[";
 for(auto pixel: {std::pair<double,double>{.5,.5},{w-.5,.5},{w-.5,h-.5},{.5,h-.5}}){
  double x=gt[0]+pixel.first*gt[1]+pixel.second*gt[2],y=gt[3]+pixel.first*gt[4]+pixel.second*gt[5];
  if(!projection->Transform(1,&x,&y))throw std::runtime_error("corner geographic transform failed");out<<"["<<x<<y<<"]";}
 out<<"]";out.release();return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
