// ============================================================================
//  ThermalDepthFusion - overlay Arducam ToF depth onto a STATIC thermal image
//
//  Thermal camera (InfiRay P2, 256x192) runs as a plain IMAGE feed (YUY2, NOT
//  temperature mode) and is the fixed base.  Arducam ToF provides BOTH a camera
//  (amplitude / IR intensity) feed and a depth feed.
//
//  One window shows: raw thermal, Arducam camera (amplitude), Arducam depth,
//  and the fused depth-on-thermal overlay.  ALL controls are on-screen buttons
//  (touch) - no keyboard needed.
//
//  Per-point calibration: align the movable depth overlay onto the thermal
//  image at a MIN and a MAX distance; at run time EACH depth pixel is placed by
//  interpolating between its MIN- and MAX-aligned position using ITS  OWN depth,
//  so every point travels along its own direction.
// ============================================================================

#include <iostream>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>

#ifndef DLLEXPORT
    #if defined(_WIN32)
        #define DLLEXPORT __declspec(dllexport)
    #else
        #define DLLEXPORT
    #endif
#endif

extern "C" {
    #include "all_config.h"
    #include "libiruvc.h"
    #include "libirparse.h"
    #include "libirprocess.h"
    #include "libirtemp.h"
    #include "thermal_cam_cmd.h"
}

#include <opencv2/opencv.hpp>
#include "ArducamTOFCamera.hpp"
using namespace Arducam;

// ---------------------------------------------------------------------------
static int TW=256, TH=192, DW=240, DH=180, RANGE_MM=4000;

// layout (window 800x480)
// main = THERMAL + Arducam CAMERA (amplitude) overlapped; right = DEPTH only
static const int MAIN_X=0, MAIN_Y=0, MAIN_W=560, MAIN_H=356;
static const int COLX=566, COLW=230;                 // right column (depth feed)
static const int F_H=200;                            // depth feed height
static const int F1Y=70;                             // depth feed y
static const int CTRL_Y=360;                         // control-bar top
static float g_msx=1.0f, g_msy=1.0f;

// ---------------------------------------------------------------------------
struct Xform { double tx=0, ty=0, s=1.0, rot=0.0; };
struct Calib {
    double min_d=300.0, max_d=1500.0;
    Xform  min_x, max_x;
    bool   flipH=false, flipV=false;
};
static Calib g_cal;
static void buildM(const Xform& x, double M[6]){
    double th=x.rot*CV_PI/180.0, c=x.s*std::cos(th), sn=x.s*std::sin(th);
    double cxd=DW/2.0, cyd=DH/2.0, cxt=TW/2.0, cyt=TH/2.0;
    M[0]=c; M[1]=-sn; M[2]=cxt+x.tx-c*cxd+sn*cyd;
    M[3]=sn;M[4]=c;   M[5]=cyt+x.ty-sn*cxd-c*cyd;
}
static inline void applyM(const double M[6],double u,double v,double&x,double&y){ x=M[0]*u+M[1]*v+M[2]; y=M[3]*u+M[4]*v+M[5]; }

// Interpolate the depth->thermal transform by scene distance (min slot .. max slot)
static Xform interpX(double d){
    double span=g_cal.max_d-g_cal.min_d; if(std::abs(span)<1e-6)span=1;
    double t=(d-g_cal.min_d)/span; t=std::max(-0.5,std::min(1.5,t));
    const Xform&a=g_cal.min_x; const Xform&b=g_cal.max_x; Xform o;
    o.tx=a.tx+t*(b.tx-a.tx); o.ty=a.ty+t*(b.ty-a.ty);
    o.s=a.s+t*(b.s-a.s); o.rot=a.rot+t*(b.rot-a.rot); return o;
}

// ---------------------------------------------------------------------------
static bool g_running=true, g_cal_mode=false, g_min_slot=true, g_cmd_capture=false;
static bool g_outline_only=false;   // show only colored depth outlines (no fill)
static bool g_show_thermal=true, g_show_depth=true;   // independent visual layer toggles
static double g_opacity=0.6; static int g_palette=2; uint8_t is_streaming=0;

// depth acquired on its own thread so the thermal loop never blocks on it
static std::mutex g_dmtx; static cv::Mat g_depth_sh, g_conf_sh;
static std::atomic<bool> g_tof_run{true}, g_depth_fresh{false};
static std::atomic<int>  g_depth_count{0};   // frames the ToF thread has grabbed

struct Palette{const char*name;int cmap;bool inv;};
static const Palette PALS[]={{"White",-1,false},{"Black",-1,true},
    {"Iron",cv::COLORMAP_INFERNO,false},{"Lava",cv::COLORMAP_HOT,false},
    {"Rainbow",cv::COLORMAP_JET,false},{"Arctic",cv::COLORMAP_WINTER,false}};
static const int NPAL=sizeof(PALS)/sizeof(PALS[0]);

// on-screen buttons -----------------------------------------------------------
enum { A_CAL,A_SLOT,A_LEFT,A_UP,A_DOWN,A_RIGHT,A_SM,A_SP,
       A_RM,A_RP,A_FH,A_FV,A_SETMIN,A_SETMAX,A_OPM,A_OPP,
       A_PAL,A_OUTLINE,A_THERM,A_DEPTHV,A_SAVE,A_LOAD,A_RESET,A_CAP,A_QUIT };
struct Btn { cv::Rect r; std::string label; int act; bool active; };
static std::vector<Btn> g_btns;
static volatile int g_action=-1;

static void signal_handler(int){ g_running=false; is_streaming=0; }

static void draw_button(cv::Mat&img,cv::Rect r,const std::string&t,cv::Scalar col,bool on){
    cv::rectangle(img,r,on?cv::Scalar(col[0]*1.7,col[1]*1.7,col[2]*1.7):col,-1);
    cv::rectangle(img,r,cv::Scalar(230,230,230),1);
    double fs=0.44; int bl=0; cv::Size ts=cv::getTextSize(t,cv::FONT_HERSHEY_SIMPLEX,fs,1,&bl);
    cv::putText(img,t,cv::Point(r.x+(r.width-ts.width)/2,r.y+(r.height+ts.height)/2),
                cv::FONT_HERSHEY_SIMPLEX,fs,cv::Scalar(255,255,255),1);
}

static void mouse_cb(int e,int x,int y,int,void*){
    if(e!=cv::EVENT_LBUTTONDOWN)return;
    for(auto&b:g_btns) if(b.r.contains(cv::Point(x,y))){ g_action=b.act; return; }
}

static cv::Mat colorize(const cv::Mat&g8){
    const Palette&p=PALS[g_palette]; cv::Mat g=g8,o;
    if(p.inv)cv::bitwise_not(g8,g);
    if(p.cmap<0)cv::cvtColor(g,o,cv::COLOR_GRAY2BGR); else cv::applyColorMap(g,o,p.cmap);
    return o;
}

// persistence -----------------------------------------------------------------
static const char* CAL_FILE="thermal_depth_calibration.json";
static bool grabnum(const std::string&s,const std::string&k,double&o){
    size_t p=s.find("\""+k+"\""); if(p==std::string::npos)return false;
    p=s.find(':',p); if(p==std::string::npos)return false; o=std::atof(s.c_str()+p+1); return true;}
static void save_cal(){
    std::ofstream f(CAL_FILE); if(!f)return; f<<std::setprecision(8);
    auto W=[&](const char*k,const Xform&x){f<<"  \""<<k<<"\": {\"tx\":"<<x.tx<<",\"ty\":"<<x.ty<<",\"s\":"<<x.s<<",\"rot\":"<<x.rot<<"}";};
    f<<"{\n  \"min_d\":"<<g_cal.min_d<<",\n  \"max_d\":"<<g_cal.max_d
     <<",\n  \"flipH\":"<<(g_cal.flipH?1:0)<<",\n  \"flipV\":"<<(g_cal.flipV?1:0)<<",\n";
    W("min_x",g_cal.min_x);f<<",\n";W("max_x",g_cal.max_x);f<<"\n}\n";
    std::cout<<"+ saved "<<CAL_FILE<<"\n";}
static void load_cal(){
    std::ifstream f(CAL_FILE); if(!f){std::cout<<"! no cal file, defaults\n";return;}
    std::stringstream ss;ss<<f.rdbuf();std::string s=ss.str();
    auto slot=[&](const std::string&o,Xform&x){size_t p=s.find("\""+o+"\"");if(p==std::string::npos)return;
        std::string sub=s.substr(p,s.find('}',p)-p+1);double v;
        if(grabnum(sub,"tx",v))x.tx=v;if(grabnum(sub,"ty",v))x.ty=v;if(grabnum(sub,"s",v))x.s=v;if(grabnum(sub,"rot",v))x.rot=v;};
    double v;if(grabnum(s,"min_d",v))g_cal.min_d=v;if(grabnum(s,"max_d",v))g_cal.max_d=v;
    if(grabnum(s,"flipH",v))g_cal.flipH=v!=0;if(grabnum(s,"flipV",v))g_cal.flipV=v!=0;
    slot("min_x",g_cal.min_x);slot("max_x",g_cal.max_x);std::cout<<"+ loaded "<<CAL_FILE<<"\n";}

static double center_depth(const cv::Mat&depth,const cv::Mat&conf,float cthr){
    int x0=DW/2-16,x1=DW/2+16,y0=DH/2-16,y1=DH/2+16;std::vector<float>v;
    for(int y=std::max(0,y0);y<std::min(DH,y1);y++)for(int x=std::max(0,x0);x<std::min(DW,x1);x++){
        float d=depth.at<float>(y,x); if(conf.at<float>(y,x)>=cthr&&d>1&&d<RANGE_MM*1.5)v.push_back(d);}
    if(v.empty())return NAN; std::nth_element(v.begin(),v.begin()+v.size()/2,v.end()); return v[v.size()/2];}

// live depth color range (mm), auto-tracked to the scene so near->far spans the
// whole colormap instead of being squished into 0..RANGE_MM.
static double g_cmin=200, g_cmax=1200;
static inline cv::Vec3b depth_color(float d){
    double span=g_cmax-g_cmin; if(span<50)span=50;
    int g=(int)std::max(0.0,std::min(255.0,(d-g_cmin)*255.0/span));
    static cv::Mat lut; if(lut.empty()){cv::Mat r(1,256,CV_8U);for(int i=0;i<256;i++)r.at<uchar>(i)=i;cv::applyColorMap(r,lut,cv::COLORMAP_JET);}
    return lut.at<cv::Vec3b>(0,g);}

// per-point scatter of depth onto thermal grid --------------------------------
// Places the DEPTH (color-by-distance) onto the thermal grid using the per-point
// depth calibration. odepth (aligned depth) is filled so the caller can draw the
// distance-colored OUTLINE at depth edges for easy thermal-overlay alignment.
static void scatter_overlay(const cv::Mat&depth,const cv::Mat&conf,bool cal_now,bool slot_min,
                            cv::Mat&ocol,cv::Mat&omask,cv::Mat&odepth){
    ocol=cv::Mat::zeros(TH,TW,CV_8UC3); omask=cv::Mat::zeros(TH,TW,CV_8U); odepth=cv::Mat::zeros(TH,TW,CV_32F);
    if(depth.empty())return;
    double Mmin[6],Mmax[6],Ms[6];
    buildM(g_cal.min_x,Mmin);buildM(g_cal.max_x,Mmax);buildM(slot_min?g_cal.min_x:g_cal.max_x,Ms);
    double span=g_cal.max_d-g_cal.min_d; if(std::abs(span)<1e-6)span=1;
    for(int v=0;v<DH;v++){
        const float* dr=depth.ptr<float>(v); const float* cr=conf.ptr<float>(v);
        double sv=g_cal.flipV?DH-1-v:v;
        for(int u=0;u<DW;u++){
            float d=dr[u]; if(cr[u]<30.f||d<=1||d>RANGE_MM*1.5)continue;
            double su=g_cal.flipH?DW-1-u:u, x,y;
            if(cal_now)applyM(Ms,su,sv,x,y);
            else{ double x0,y0,x1,y1;applyM(Mmin,su,sv,x0,y0);applyM(Mmax,su,sv,x1,y1);
                  double t=(d-g_cal.min_d)/span; t=std::max(-0.5,std::min(1.5,t));
                  x=x0+t*(x1-x0);y=y0+t*(y1-y0);}
            int xi=(int)std::lround(x),yi=(int)std::lround(y);
            if(xi<0||xi>=TW||yi<0||yi>=TH)continue;
            ocol.at<cv::Vec3b>(yi,xi)=depth_color(d); omask.at<uchar>(yi,xi)=255; odepth.at<float>(yi,xi)=d;
        }
    }
    // fill the small gaps left by forward scatter (depth 240x180 -> thermal grid)
    static cv::Mat kern=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    cv::dilate(ocol,ocol,kern); cv::dilate(omask,omask,kern); cv::dilate(odepth,odepth,kern);
}

// build the button bar (fixed positions, active states vary) ------------------
static void layout_buttons(){
    g_btns.clear();
    std::vector<std::vector<Btn>> rows;
    auto A=[&](std::vector<Btn>&row,const std::string&l,int act,bool on=false){ row.push_back({cv::Rect(),l,act,on}); };
    std::vector<Btn> r0,r1,r2;
    A(r0,"Cal",A_CAL,g_cal_mode); A(r0,g_min_slot?"MIN":"MAX",A_SLOT,true);
    A(r0,"<",A_LEFT); A(r0,"Up",A_UP); A(r0,"Dn",A_DOWN); A(r0,">",A_RIGHT);
    A(r0,"Sc-",A_SM); A(r0,"Sc+",A_SP);
    A(r1,"Rot-",A_RM); A(r1,"Rot+",A_RP); A(r1,"FlipH",A_FH,g_cal.flipH); A(r1,"FlipV",A_FV,g_cal.flipV);
    A(r1,"SetMin",A_SETMIN); A(r1,"SetMax",A_SETMAX); A(r1,"Op-",A_OPM); A(r1,"Op+",A_OPP);
    A(r2,"Palette",A_PAL); A(r2,"Outline",A_OUTLINE,g_outline_only);
    A(r2,"Therm",A_THERM,g_show_thermal); A(r2,"Depth",A_DEPTHV,g_show_depth);
    A(r2,"Save",A_SAVE); A(r2,"Load",A_LOAD); A(r2,"Reset",A_RESET);
    A(r2,"Capture",A_CAP); A(r2,"Quit",A_QUIT);
    rows={r0,r1,r2};
    int margin=4,gap=3,rowH=36,rowGap=4;
    for(size_t ri=0;ri<rows.size();ri++){
        int n=rows[ri].size(); int cw=(800-2*margin-(n-1)*gap)/n;
        int y=CTRL_Y + (int)ri*(rowH+rowGap);
        for(int i=0;i<n;i++){ Btn b=rows[ri][i]; b.r=cv::Rect(margin+i*(cw+gap),y,cw,rowH); g_btns.push_back(b); }
    }
}

static double live_d=NAN;
static void apply_action(int act){
    Xform* X=g_min_slot?&g_cal.min_x:&g_cal.max_x;
    const double MOVE=2.0,SCST=0.02,ROT=1.0;
    bool adj=(act>=A_LEFT&&act<=A_RP)||act==A_FH||act==A_FV;
    if(adj) g_cal_mode=true;      // show the slot you're editing
    switch(act){
        case A_CAL: g_cal_mode=!g_cal_mode; break;
        case A_SLOT: g_min_slot=!g_min_slot; break;
        case A_LEFT: X->tx-=MOVE; break;   case A_RIGHT:X->tx+=MOVE; break;
        case A_UP:   X->ty-=MOVE; break;   case A_DOWN: X->ty+=MOVE; break;
        case A_SM:   X->s=std::max(0.05,X->s-SCST); break; case A_SP: X->s+=SCST; break;
        case A_RM:   X->rot-=ROT; break;   case A_RP:   X->rot+=ROT; break;
        case A_FH:   g_cal.flipH=!g_cal.flipH; break; case A_FV: g_cal.flipV=!g_cal.flipV; break;
        case A_SETMIN: if(!std::isnan(live_d)){g_cal.min_d=live_d;std::cout<<"+ min="<<(int)live_d<<"mm\n";} break;
        case A_SETMAX: if(!std::isnan(live_d)){g_cal.max_d=live_d;std::cout<<"+ max="<<(int)live_d<<"mm\n";} break;
        case A_OPM: g_opacity=std::max(0.0,g_opacity-0.05); break;
        case A_OPP: g_opacity=std::min(1.0,g_opacity+0.05); break;
        case A_PAL: g_palette=(g_palette+1)%NPAL; break;
        case A_OUTLINE: g_outline_only=!g_outline_only; break;
        case A_THERM:   g_show_thermal=!g_show_thermal; break;
        case A_DEPTHV:  g_show_depth=!g_show_depth; break;
        case A_SAVE:save_cal(); break; case A_LOAD:load_cal(); break;
        case A_RESET:*X=Xform{0,0,(double)TW/DW,0}; break;
        case A_CAP: g_cmd_capture=true; break;
        case A_QUIT:g_running=false; break;
    }
}

// y16_preview_start returns 0 but doesn't always actually switch the stream to
// Y16 - sometimes it stays YUY2 (odd byte == 128), and YUY2 dies at the camera's
// ~10s auto-shutter (uvc_frame_get -> ret=-12). Real Y16 has varying odd bytes
// (~74). Retry the switch until a frame proves Y16 engaged. Returns true if so.
static bool engage_y16(void* fbuf){
    for(int attempt=0; attempt<6; attempt++){
        y16_preview_stop(PREVIEW_PATH0); usleep(200000);
        set_prop_tpd_params(TPD_PROP_GAIN_SEL,1);
        if(y16_preview_start(PREVIEW_PATH0,Y16_MODE_TEMPERATURE)!=0){ usleep(300000); continue; }
        usleep(600000);
        for(int k=0;k<15;k++){
            if(uvc_frame_get(fbuf)==0){
                uint8_t* rb=(uint8_t*)fbuf;
                if(rb[1]!=128 || rb[3]!=128 || rb[5]!=128){
                    printf("+ Y16 engaged on attempt %d (odd byte=%d)\n",attempt+1,rb[1]); fflush(stdout);
                    return true;
                }
                break;   // got a frame but it's still YUY2 -> retry the switch
            }
            usleep(40000);
        }
    }
    return false;
}

int main(){
    signal(SIGINT,signal_handler); signal(SIGTERM,signal_handler);
    std::cout<<"===== ThermalDepthFusion (image mode, touch UI) =====\n";

    // thermal (image / YUY2)
    #define CK(m) do{fprintf(stderr,"[ck] %s\n",m);fflush(stderr);}while(0)
    CK("vdcmd_init"); vdcmd_init();
    CK("uvc_camera_init"); if(uvc_camera_init()<0){std::cerr<<"x thermal init failed\n";return -1;}
    DevCfg_t devs[64]; memset(devs,0,sizeof(devs));
    CK("uvc_camera_list"); if(uvc_camera_list(devs)<0)return -1;
    int ti=-1; for(int i=0;i<10&&devs[i].vid;i++) if(devs[i].vid==0x0BDA&&devs[i].pid==0x5840){ti=i;break;}
    if(ti<0){std::cerr<<"x thermal camera not found\n";return -1;}
    CameraStreamInfo_t streams[32]; uvc_camera_info_get(devs[ti],streams);
    CameraParam_t cp; memset(&cp,0,sizeof(cp));
    cp.dev_cfg=devs[ti];cp.format=streams[0].format;cp.width=streams[0].width;cp.height=streams[0].height;
    cp.frame_size=cp.width*cp.height*2;cp.fps=streams[0].fps[0];cp.timeout_ms_delay=1000;
    TW=cp.width;TH=cp.height; g_msx=(float)MAIN_W/TW; g_msy=(float)MAIN_H/TH;
    CK("uvc_camera_open"); if(uvc_camera_open(devs[ti])<0)return -1;
    void* fbuf=uvc_frame_buf_create(cp);
    CK("stream_start"); if(uvc_camera_stream_start(cp,nullptr)<0)return -1;
    is_streaming=1;
    usleep(500000);
    // Configure the image ISP and warm the pipeline up the way the stable apps do,
    // then drop back to the plain YUY2 image path. Running the stream with no ISP
    // setup is what leaves the SDK control channel wedging/stalling. This briefly
    // touches Y16 during warmup but ends in IMAGE mode (temperature mode NOT kept on).
    set_prop_tpd_params(TPD_PROP_GAIN_SEL,1);
    set_prop_image_params(IMAGE_PROP_MODE_AGC,3);
    set_prop_image_params(IMAGE_PROP_LEVEL_DDE,2);
    set_prop_image_params(IMAGE_PROP_LEVEL_SNR,2);
    set_prop_image_params(IMAGE_PROP_LEVEL_TNR,2);
    set_prop_image_params(IMAGE_PROP_LEVEL_BRIGHTNESS,140);
    set_prop_image_params(IMAGE_PROP_LEVEL_CONTRAST,150);
    // The camera's periodic auto-shutter/FFC (~10s) is what kills the default image
    // stream (uvc_frame_get -> ret=-12). Disable it, and take ONE manual FFC now for
    // a clean baseline. This keeps the reliable YUY2 image stream alive indefinitely
    // without any Y16/temperature mode.
    set_prop_auto_shutter_params(SHUTTER_PROP_SWITCH, 0);       // no periodic auto-FFC
    set_prop_auto_shutter_params(SHUTTER_PROP_MAX_INTERVAL, 65535); // and if it ever fires, make it rare
    ooc_b_update(OOC_B_UPDATE);
    usleep(300000);
    std::cout<<"+ thermal "<<TW<<"x"<<TH<<" @ "<<cp.fps<<" fps (auto-shutter OFF)\n";

    // Arducam ToF (depth + amplitude/confidence camera).
    // The unicam capture node (/dev/videoN) can renumber across reboots, so find
    // the "unicam-image" node and pass its index to open() instead of assuming 0.
    int csi_idx=0;
    for(int n=0;n<32;n++){
        char p[64]; snprintf(p,sizeof(p),"/sys/class/video4linux/video%d/name",n);
        std::ifstream nf(p); if(!nf)continue; std::string nm; std::getline(nf,nm);
        if(nm.find("unicam")!=std::string::npos){ csi_idx=n; break; }
    }
    std::cout<<"+ ToF CSI node index "<<csi_idx<<" (/dev/video"<<csi_idx<<")\n";
    ArducamTOFCamera tof;
    if(tof.open(Connection::CSI,csi_idx)){std::cerr<<"x ToF open failed (/dev/video"<<csi_idx<<")\n";return -1;}
    if(tof.start(FrameType::DEPTH_FRAME)){std::cerr<<"x ToF start failed\n";return -1;}
    tof.setControl(Control::RANGE,RANGE_MM);
    auto tinfo=tof.getCameraInfo(); DW=tinfo.width; DH=tinfo.height;
    std::cout<<"+ ToF "<<DW<<"x"<<DH<<" (depth + amplitude) range "<<RANGE_MM<<"mm\n";

    double defScale=(double)TW/DW; g_cal.min_x={0,0,defScale,0}; g_cal.max_x={0,0,defScale,0};
    load_cal();

    cv::namedWindow("ThermalDepthFusion",cv::WINDOW_NORMAL);
    cv::setWindowProperty("ThermalDepthFusion",cv::WND_PROP_FULLSCREEN,cv::WINDOW_FULLSCREEN);
    cv::setMouseCallback("ThermalDepthFusion",mouse_cb,nullptr);
    std::cout<<"+ touch UI ready\n";

    // ToF acquisition thread: continuously grab depth+confidence and publish the
    // latest copy. Keeps ALL Arducam SDK calls on one thread and means the thermal
    // loop below never blocks waiting on depth (fixes the stall/freeze).
    std::thread tof_thread([&](){
        while(g_tof_run.load()){
            ArducamFrameBuffer* fr=tof.requestFrame(200);
            if(!fr){ usleep(3000); continue; }   // avoid pegging a core when no frame
            Arducam::FrameFormat ff; fr->getFormat(FrameType::DEPTH_FRAME,ff);
            float* dp=(float*)fr->getData(FrameType::DEPTH_FRAME);
            float* cptr=(float*)fr->getData(FrameType::CONFIDENCE_FRAME);
            if(dp&&cptr){
                cv::Mat d,c; cv::Mat(ff.height,ff.width,CV_32F,dp).copyTo(d);
                cv::Mat(ff.height,ff.width,CV_32F,cptr).copyTo(c);
                { std::lock_guard<std::mutex> lk(g_dmtx); d.copyTo(g_depth_sh); c.copyTo(g_conf_sh); }
                g_depth_fresh.store(true); g_depth_count.fetch_add(1);
            }
            tof.releaseFrame(fr);
        }
    });

    // Start the Y16 image path right before the loop. The DEFAULT stream dies ~10s
    // in (camera auto-shutter/FFC; ret=-12) but the Y16 path survives it (this is
    // the mode ThermalViewer runs stably). Rendered purely as a normalized image,
    // no temperatures shown. (Clean start is guaranteed: `make run` kills stale
    // instances + resets the cam first.)
    // Default YUY2 image stream: reliable to start, decodes via raw[i*2], and with
    // auto-shutter disabled above it does NOT die at ~10s. No temperature mode.
    bool y16=false;
    printf("+ thermal image ready (YUY2 luma, auto-shutter off, no temps)\n"); fflush(stdout);

    // Silence ONLY the SDK's [WARN] flood (stderr); keep stdout for the heartbeat
    // so we can see which side stalls if it freezes.
    { int dn=open("/dev/null",O_WRONLY); if(dn>=0){ dup2(dn,2); if(dn>2)close(dn); } }

    cv::Mat depth_f,conf_f; int fc=0, uvc_fail=0, last_ret=0, consec_fail=0; std::string g_toast; int toast_frames=0;
    auto hbt=std::chrono::steady_clock::now(); int lfc=0,ldc=0,lfail=0;
    while(g_running&&is_streaming){
        // heartbeat at TOP so it prints even when frames fail
        { auto now=std::chrono::steady_clock::now();
          if(now-hbt>=std::chrono::seconds(1)){ int dc=g_depth_count.load();
            printf("hb: thermal=%d depth=%d uvcfail=%d(ret=%d) live=%dmm crange=%.0f-%.0f %s\n",
                   fc-lfc, dc-ldc, uvc_fail-lfail, last_ret,
                   std::isnan(live_d)?-1:(int)live_d, g_cmin, g_cmax, g_cal_mode?"CAL":"RUN");
            fflush(stdout); lfc=fc; ldc=dc; lfail=uvc_fail; hbt=now; }
        }
        int r=uvc_frame_get(fbuf); last_ret=r;
        if(r!=0){ uvc_fail++;
            // ret=-12: the thermal stream died (shutter/FFC the SDK couldn't ride out).
            // Auto-recover with a full stream restart so it self-heals instead of freezing.
            if(++consec_fail>=5){ printf("! thermal ret=%d - restarting stream...\n",r); fflush(stdout);
                uvc_camera_stream_close(KEEP_CAM_SIDE_PREVIEW); usleep(300000);
                uvc_frame_buf_release(fbuf); fbuf=uvc_frame_buf_create(cp);
                if(uvc_camera_stream_start(cp,nullptr)==0){ usleep(300000);
                    set_prop_auto_shutter_params(SHUTTER_PROP_SWITCH,0);
                    set_prop_auto_shutter_params(SHUTTER_PROP_MAX_INTERVAL,65535);
                    ooc_b_update(OOC_B_UPDATE);
                    printf("+ thermal stream restarted\n"); fflush(stdout); }
                consec_fail=0;
            }
            usleep(8000); continue;
        }
        consec_fail=0;
        fc++;
        if(fc==1){   // one-time frame-format diagnostic
            uint8_t* rb=(uint8_t*)fbuf; int N=TW*TH*2;
            int bmin=255,bmax=0; long bsum=0;
            for(int i=0;i<N;i++){int b=rb[i]; if(b<bmin)bmin=b; if(b>bmax)bmax=b; bsum+=b;}
            uint16_t* w=(uint16_t*)fbuf; int wmin=65535,wmax=0; long wsum=0;
            for(int i=0;i<TW*TH;i++){int v=w[i]; if(v<wmin)wmin=v; if(v>wmax)wmax=v; wsum+=v;}
            int emin=255,emax=0,omin=255,omax=0;
            for(int i=0;i<TW*TH;i++){int e=rb[i*2],o=rb[i*2+1];
                if(e<emin)emin=e; if(e>emax)emax=e; if(o<omin)omin=o; if(o>omax)omax=o;}
            printf("FMT: bytes[%d..%d mean=%ld] u16[%d..%d mean=%ld] even[%d..%d] odd[%d..%d] first16=",
                   bmin,bmax,bsum/N,wmin,wmax,wsum/(TW*TH),emin,emax,omin,omax);
            for(int i=0;i<16;i++)printf("%02x ",rb[i]); printf("\n"); fflush(stdout);
        }
        cv::Mat tgray;
        if(y16){ cv::Mat m16(TH,TW,CV_16UC1,(uint16_t*)fbuf); cv::normalize(m16,tgray,0,255,cv::NORM_MINMAX,CV_8UC1); }
        else { tgray=cv::Mat(TH,TW,CV_8UC1); uint8_t* raw=(uint8_t*)fbuf; for(int i=0;i<TW*TH;i++)tgray.data[i]=raw[i*2]; }
        cv::Mat thermal=colorize(tgray);   // normalized grayscale -> palette (image only, no temps)

        // grab the latest depth the ToF thread published (non-blocking)
        if(g_depth_fresh.exchange(false)){
            std::lock_guard<std::mutex> lk(g_dmtx);
            g_depth_sh.copyTo(depth_f); g_conf_sh.copyTo(conf_f);
        }
        if(!depth_f.empty()){
            live_d=center_depth(depth_f,conf_f,30.f);
            // auto-track color range to the ROBUST valid-depth spread (5th..95th
            // percentile) so a few noise/background outliers don't blow it out.
            std::vector<float> vv; vv.reserve(DW*DH/2);
            for(int y=0;y<DH;y++){ const float* dp=depth_f.ptr<float>(y); const float* cp=conf_f.ptr<float>(y);
                for(int x=0;x<DW;x+=2){ float d=dp[x];
                    if(cp[x]>=30.f && d>150.f && d<RANGE_MM*0.98f) vv.push_back(d); } }
            if(vv.size()>100){
                size_t lo=vv.size()*5/100, hi=vv.size()*95/100;
                std::nth_element(vv.begin(),vv.begin()+lo,vv.end()); float p5=vv[lo];
                std::nth_element(vv.begin(),vv.begin()+hi,vv.end()); float p95=vv[hi];
                if(p95-p5<120){ float mid=(p5+p95)/2; p5=mid-60; p95=mid+60; }
                g_cmin=0.85*g_cmin+0.15*p5; g_cmax=0.85*g_cmax+0.15*p95;   // smoothed
            }
        }

        // raw arducam camera image (amplitude) for the separate feed
        cv::Mat amp8;
        if(!conf_f.empty()) cv::normalize(conf_f,amp8,0,255,cv::NORM_MINMAX,CV_8U);

        // ---- temporal smoothing of depth (kills the random pixel flicker) ----
        static cv::Mat g_dema;
        if(!depth_f.empty()){
            if(g_dema.empty()||g_dema.size()!=depth_f.size()) depth_f.copyTo(g_dema);
            else { cv::Mat valid=(conf_f>=40.f)&(depth_f>150.f);
                   cv::Mat mix; cv::addWeighted(g_dema,0.6,depth_f,0.4,0,mix); mix.copyTo(g_dema,valid); }
        }

        // ---- clean depth overlay via DENSE warp at the scene transform ----
        // (replaces the per-point forward scatter, which was holey/noisy). One affine
        // warp of the whole depth image -> a solid region; outline = its silhouette.
        cv::Mat base = g_show_thermal ? thermal : cv::Mat::zeros(thermal.size(),thermal.type());
        cv::Mat omask, odepth; cv::Mat fused=base.clone();
        if(!g_dema.empty()){
            cv::Mat ds=g_dema.clone(), cs=conf_f.clone();
            if(g_cal.flipH){ cv::flip(ds,ds,1); cv::flip(cs,cs,1); }
            if(g_cal.flipV){ cv::flip(ds,ds,0); cv::flip(cs,cs,0); }
            Xform X = g_cal_mode ? (g_min_slot?g_cal.min_x:g_cal.max_x)
                                 : interpX(std::isnan(live_d)?(g_cal.min_d+g_cal.max_d)/2:live_d);
            double M6[6]; buildM(X,M6);
            cv::Mat M=(cv::Mat_<double>(2,3)<<M6[0],M6[1],M6[2],M6[3],M6[4],M6[5]);
            cv::Mat wdepth,wconf;
            cv::warpAffine(ds,wdepth,M,cv::Size(TW,TH),cv::INTER_NEAREST,cv::BORDER_CONSTANT,cv::Scalar(0));
            cv::warpAffine(cs,wconf,M,cv::Size(TW,TH),cv::INTER_NEAREST,cv::BORDER_CONSTANT,cv::Scalar(0));
            cv::Mat vmask=(wconf>=40)&(wdepth>150)&(wdepth<(float)RANGE_MM*0.98f);
            static cv::Mat k3=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
            cv::morphologyEx(vmask,vmask,cv::MORPH_OPEN,k3);    // drop speckle
            cv::morphologyEx(vmask,vmask,cv::MORPH_CLOSE,k3);   // fill pinholes
            double span=g_cmax-g_cmin; if(span<50)span=50;
            cv::Mat d8; wdepth.convertTo(d8,CV_8U,255.0/span,-g_cmin*255.0/span);
            cv::Mat ocol; cv::applyColorMap(d8,ocol,cv::COLORMAP_JET);
            cv::Mat sil; cv::morphologyEx(vmask,sil,cv::MORPH_GRADIENT,k3);   // clean object silhouette
            cv::Mat edg; cv::Canny(d8,edg,50,140); edg&=vmask;               // internal depth edges
            cv::Mat outline=sil|edg; cv::dilate(outline,outline,k3);
            omask=vmask; wdepth.copyTo(odepth); odepth.setTo(0,~vmask);       // always kept for capture
            if(g_show_depth){   // DRAW the depth layer (view toggle only; streaming unaffected)
                double op = g_show_thermal ? g_opacity : 1.0;   // full depth when thermal hidden
                if(!g_outline_only){ cv::Mat bl; cv::addWeighted(base,1.0-op,ocol,op,0,bl); bl.copyTo(fused,vmask); }
                ocol.copyTo(fused,outline);                                  // distance-colored outline
            }
        }

        cv::Mat ui(480,800,CV_8UC3,cv::Scalar(22,22,22));
        // main overlay
        cv::Mat mv; cv::resize(fused,mv,cv::Size(MAIN_W,MAIN_H)); mv.copyTo(ui(cv::Rect(MAIN_X,MAIN_Y,MAIN_W,MAIN_H)));
        cv::drawMarker(ui,cv::Point(MAIN_X+MAIN_W/2,MAIN_Y+MAIN_H/2),cv::Scalar(0,255,255),cv::MARKER_CROSS,14,1);

        auto feed=[&](const cv::Mat& img,int y,const char* lbl,cv::Scalar lc){
            cv::Mat r; if(img.empty())r=cv::Mat::zeros(F_H,COLW,CV_8UC3); else cv::resize(img,r,cv::Size(COLW,F_H));
            r.copyTo(ui(cv::Rect(COLX,y,COLW,F_H)));
            cv::rectangle(ui,cv::Rect(COLX,y,COLW,F_H),cv::Scalar(80,80,80),1);
            cv::putText(ui,lbl,cv::Point(COLX+4,y+15),cv::FONT_HERSHEY_SIMPLEX,0.42,lc,1);
        };
        // RAW ARDUCAM feed (amplitude) separate - the depth is now the outline
        // overlay on the thermal in the main pane above.
        cv::Mat rawcam;
        if(!amp8.empty()) cv::cvtColor(amp8,rawcam,cv::COLOR_GRAY2BGR);
        feed(rawcam,F1Y,"ARDUCAM",cv::Scalar(200,200,0));

        // HUD
        std::stringstream ss;
        cv::putText(ui,g_cal_mode?(g_min_slot?"CAL: MIN slot":"CAL: MAX slot"):"RUN (per-point)",
                    cv::Point(8,18),cv::FONT_HERSHEY_SIMPLEX,0.5,g_cal_mode?cv::Scalar(0,220,255):cv::Scalar(0,255,120),2);
        ss<<"dist "<<(std::isnan(live_d)?"--":std::to_string((int)live_d))<<"mm  min "<<(int)g_cal.min_d
          <<"  max "<<(int)g_cal.max_d<<"  op "<<std::fixed<<std::setprecision(2)<<g_opacity;
        cv::putText(ui,ss.str(),cv::Point(8,MAIN_H-10),cv::FONT_HERSHEY_SIMPLEX,0.44,cv::Scalar(235,235,235),1);
        if(g_cal_mode){ const Xform&X=g_min_slot?g_cal.min_x:g_cal.max_x; ss.str("");
            ss<<"tx"<<(int)X.tx<<" ty"<<(int)X.ty<<" s"<<std::setprecision(2)<<X.s<<" r"<<std::setprecision(1)<<X.rot;
            cv::putText(ui,ss.str(),cv::Point(8,MAIN_H-30),cv::FONT_HERSHEY_SIMPLEX,0.44,cv::Scalar(255,220,120),1); }

        layout_buttons();
        for(auto&b:g_btns){
            cv::Scalar c=cv::Scalar(70,70,70);
            if(b.act==A_CAP)c=cv::Scalar(40,140,40); else if(b.act==A_QUIT)c=cv::Scalar(160,40,40);
            else if(b.act==A_SLOT)c=cv::Scalar(90,70,20); else if(b.act>=A_LEFT&&b.act<=A_RP)c=cv::Scalar(55,55,95);
            draw_button(ui,b.r,b.label,c,b.active);
        }

        if(g_cmd_capture){ g_cmd_capture=false;
            auto t=std::time(nullptr);auto lt=*std::localtime(&t);char st[32];std::strftime(st,sizeof(st),"%Y%m%d_%H%M%S",&lt);
            std::string b=std::string("fusion_")+st;
            cv::imwrite(b+"_thermal.png",thermal); cv::imwrite(b+"_overlay.png",fused);
            if(!amp8.empty())cv::imwrite(b+"_arducam_cam.png",amp8);
            {std::ofstream f(b+"_depth_f32_"+std::to_string(TW)+"x"+std::to_string(TH)+".raw",std::ios::binary);
             if(!odepth.empty())f.write((char*)odepth.data,TW*TH*sizeof(float));}
            {std::ofstream f(b+"_meta.json");
             f<<"{\n \"stamp\":\""<<st<<"\",\n \"thermal_wh\":["<<TW<<","<<TH<<"],\n \"depth_src_wh\":["<<DW<<","<<DH<<"],\n"
              <<" \"range_mm\":"<<RANGE_MM<<",\n \"live_dist_mm\":"<<(std::isnan(live_d)?0:live_d)<<",\n"
              <<" \"min_d\":"<<g_cal.min_d<<", \"max_d\":"<<g_cal.max_d<<"\n}\n";}
            g_toast="Saved "+b; toast_frames=60;
        }

        // on-screen toast (stdout is redirected, so show feedback here)
        if(toast_frames>0){ toast_frames--;
            cv::rectangle(ui,cv::Rect(MAIN_X+6,MAIN_H-56,MAIN_W-12,22),cv::Scalar(20,90,20),-1);
            cv::putText(ui,g_toast,cv::Point(MAIN_X+12,MAIN_H-40),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(255,255,255),1);
        }

        cv::imshow("ThermalDepthFusion",ui);
        cv::waitKey(1);
        if(g_action>=0){ int a=g_action; g_action=-1; apply_action(a); }
    }

    g_tof_run.store(false); if(tof_thread.joinable()) tof_thread.join();
    if(y16) y16_preview_stop(PREVIEW_PATH0);
    tof.stop(); tof.close(); cv::destroyAllWindows();
    uvc_camera_stream_close(KEEP_CAM_SIDE_PREVIEW); uvc_frame_buf_release(fbuf);
    uvc_camera_close(); uvc_camera_release();
    std::cout<<"+ frames "<<fc<<" clean exit\n"; return 0;
}
