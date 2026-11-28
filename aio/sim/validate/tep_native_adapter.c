
/* AIOGym2 standalone wrapper. Upstream equations above are unmodified.
   Each handle owns model, RNG, analyzer and PI states; no global plant state.
   Physics: RK4, one-second measurement clock, disturbance/RNG changes are
   committed on the first stage only. Later stages operate on copies.
   Outputs/measurement noise are computed once at each integer second.
*/
#define API __declspec(dllexport)
typedef struct {
    struct stModelData m;
    double y[50], seconds, cmd[12], initial[41], integ[7];
    double gas_source, product_source, gas_capture, product_capture;
    int stopped;
} TEHandle;

API int te_size(void) { return (int)sizeof(TEHandle); }
API void *te_new(double seed) {
    TEHandle *h=(TEHandle*)calloc(1,sizeof(TEHandle));
    double t=0,dy[50],flags=32+64; /* independent RNG streams, revised disturbance clock */
    teinit(&h->m,&NX,&t,h->y,dy,&seed,&flags);
    memcpy(h->cmd,h->m.pv_.xmv,sizeof(h->cmd));
    memcpy(h->initial,h->m.pv_.xmeas,sizeof(h->initial));
    return h;
}
API void te_free(void *v){free(v);}
API void te_save(void *v,void *b){memcpy(b,v,sizeof(TEHandle));}
API void te_load(void *v,void *b){memcpy(v,b,sizeof(TEHandle));}

static double clampte(double x,double lo,double hi){return fmin(hi,fmax(lo,x));}
static void pi_controls(TEHandle *h,const double *offset){
    /* Independent engineering baseline, not claimed to reproduce Ricker's
       17-loop scheme. Incremental, conditional-integration PI at 1 second.
       Public measurements only. Seven setpoint trims, frozen gains. */
    const int yi[7]={6,7,8,10,11,14,17};
    const double gains[7]={-.04,.30,-2.0,-2.0,-1.0,-1.0,1.0};
    const double tis[7]={1200,1800,300,600,300,300,600};
    const double scales[7]={100,5,2,5,5,5,5};
    const double bases[7]={40.06374673,0,41.10581288,18.11349055,38.1003437,46.53415582,47.44573456};
    double val[7];
    for(int j=0;j<7;j++){
        double error=h->initial[yi[j]]+scales[j]*offset[j]-h->m.pv_.xmeas[yi[j]];
        double p=gains[j]*error;
        double next=h->integ[j]+p/tis[j];
        double base=bases[j];
        const int vi[7]={5,-1,9,10,6,7,8};
        if(j==1){
            if((p+next<15 && p+next>-15) || (p+next>=15 && p<0) || (p+next<=-15 && p>0))h->integ[j]=next;
            val[j]=clampte(p+h->integ[j],-15,15);
        }else{
            double request=base+p+next;
            if((request<100 && request>0)||(request>=100&&p<0)||(request<=0&&p>0))h->integ[j]=next;
            val[j]=clampte(base+p+h->integ[j],0,100);
            h->cmd[vi[j]]=val[j];
        }
    }
    const double feeds[4]={63.05263039,53.97970677,24.64355755,61.30192144};
    for(int j=0;j<4;j++)h->cmd[j]=clampte(feeds[j]*(1+val[1]/100),0,100);
    h->cmd[4]=22.21;h->cmd[11]=50;
}
static int secondte(TEHandle*h,const double*input,int supervisory,int nsub,int fault){
    if(h->stopped)return h->stopped;
    if(supervisory)pi_controls(h,input);
    else for(int j=0;j<12;j++)h->cmd[j]=clampte(input[j],0,100);
    memcpy(h->m.pv_.xmv,h->cmd,sizeof(h->cmd));
    memset(h->m.dvec_.idv,0,28*sizeof(double));
    if(fault>0 && fault<=20)h->m.dvec_.idv[fault-1]=1;
    const double dt=1.0/(3600*nsub);
    for(int s=0;s<nsub;s++){
        double t=(h->seconds+(double)s/nsub)/3600;
        double k1[50],k2[50],k3[50],k4[50],tmp[50],t2=t+dt/2,t4=t+dt;
        h->m.tlastcomp=-1;
        tefunc(&h->m,&NX,&t,h->y,k1,2);
        struct stModelData m2=h->m,m3=h->m,m4=h->m;
        for(int j=0;j<50;j++)tmp[j]=h->y[j]+dt*k1[j]/2;
        m2.tlastcomp=-1;tefunc(&m2,&NX,&t2,tmp,k2,2);
        for(int j=0;j<50;j++)tmp[j]=h->y[j]+dt*k2[j]/2;
        m3.tlastcomp=-1;tefunc(&m3,&NX,&t2,tmp,k3,2);
        for(int j=0;j<50;j++)tmp[j]=h->y[j]+dt*k3[j];
        m4.tlastcomp=-1;tefunc(&m4,&NX,&t4,tmp,k4,2);
        for(int j=0;j<50;j++){
            h->y[j]+=dt*(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6;
            if(!isfinite(h->y[j])){h->stopped=99;return 99;}
        }
    }
    h->seconds++;
    double t=h->seconds/3600,dy[50];
    double oldgas=h->m.teproc_.tgas,oldprod=h->m.teproc_.tprod;
    tefunc(&h->m,&NX,&t,h->y,dy,1);
    if(h->m.teproc_.tgas!=oldgas){h->gas_source=h->gas_capture;h->gas_capture=h->seconds;}
    if(h->m.teproc_.tprod!=oldprod){h->product_source=h->product_capture;h->product_capture=h->seconds;}
    h->stopped=(int)h->m.dvec_.idv[28];
    for(int j=0;j<41;j++)if(!isfinite(h->m.pv_.xmeas[j]))h->stopped=99;
    return h->stopped;
}
API int te_advance(void*v,const double*input,int supervisory,int seconds,int nsub,int fault){
    TEHandle*h=v;
    for(int k=0;k<seconds;k++){int code=secondte(h,input,supervisory,nsub,fault);if(code)return code;}
    return 0;
}
API void te_read(void*v,double*out){
    TEHandle*h=v;
    memcpy(out,h->m.pv_.xmeas,41*sizeof(double));
    memcpy(out+41,h->y+38,12*sizeof(double));
    memcpy(out+53,h->cmd,12*sizeof(double));
    out[65]=h->seconds;out[66]=h->seconds-h->gas_source;out[67]=h->seconds-h->product_source;
    /* Evaluation-only true product G,H mol%, volume m3/h. Never actor inputs. */
    out[68]=100*h->m.teproc_.xst[102];out[69]=100*h->m.teproc_.xst[103];
    out[70]=h->m.teproc_.ftm[12]/h->m.teproc_.dlc/35.3145;
    out[71]=h->stopped;
}
