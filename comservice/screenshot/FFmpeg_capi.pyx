#!/usr/bin/env python3
# coding=utf-8

from ctypes import *
import sys
import os
###
#对于项目，需要使用到获取系统可用的音频设备信息，录制音频设备输出到文件或者 pipe中即可
#
###

#设置ctype的类型转换定义
ctypedef signed char int8_t
ctypedef unsigned char uint8_t
ctypedef signed short int16_t
ctypedef unsigned short uint16_t
ctypedef signed long int32_t
ctypedef unsigned long uint32_t
ctypedef signed long long int64_t
ctypedef unsigned long long uint64_t
# other types
#ctypedef char const char "const char"
#ctypedef AVCodecContext const AVCodecContext "const AVCodecContext"
ctypedef struct const_struct_AVSubtitle "const struct AVSubtitle"
#ctypedef AVFrame const AVFrame "const AVFrame" 
#ctypedef AVClass const AVClass "const AVClass"
ctypedef struct const_struct_AVCodec "const struct AVCodec"
#ctypedef AVCodecDescriptor const AVCodecDescriptor "const AVCodecDescriptor"
#ctypedef AVPacket const AVPacket "const AVPacket"
ctypedef struct const_struct_AVHWAccel "const struct AVHWAccel"
ctypedef struct const_struct_AVProfile "const struct AVProfile"
ctypedef struct const_struct_AVOption "const struct AVOption"
ctypedef struct const_struct_AVClass "const struct AVClass"

#解析定义用到的头文件和相关的定义
#opt.h
cdef extern from "../libs/ffmpeg/include/libavutil/opt.h":
    struct AVOptionRange:
        const char *str
        double value_min 
        double value_max
        double component_min
        double component_max
        int is_range
        
    struct AVOptionRanges:
        AVOptionRange **range   #< Array of option ranges
        int nb_ranges           #< Number of ranges per component
        int nb_components       #< Number of componentes
#log.h
cdef extern from "../libs/ffmpeg/include/libavutil/log.h":
    enum AVClassCategory:
        AV_CLASS_CATEGORY_NA = 0,
        AV_CLASS_CATEGORY_INPUT,
        AV_CLASS_CATEGORY_OUTPUT,
        AV_CLASS_CATEGORY_MUXER,
        AV_CLASS_CATEGORY_DEMUXER,
        AV_CLASS_CATEGORY_ENCODER,
        AV_CLASS_CATEGORY_DECODER,
        AV_CLASS_CATEGORY_FILTER,
        AV_CLASS_CATEGORY_BITSTREAM_FILTER,
        AV_CLASS_CATEGORY_SWSCALER,
        AV_CLASS_CATEGORY_SWRESAMPLER,
        AV_CLASS_CATEGORY_DEVICE_VIDEO_OUTPUT = 40,
        AV_CLASS_CATEGORY_DEVICE_VIDEO_INPUT,
        AV_CLASS_CATEGORY_DEVICE_AUDIO_OUTPUT,
        AV_CLASS_CATEGORY_DEVICE_AUDIO_INPUT,
        AV_CLASS_CATEGORY_DEVICE_OUTPUT,
        AV_CLASS_CATEGORY_DEVICE_INPUT,
        AV_CLASS_CATEGORY_NB
    struct AVClass:
        const char* class_name
        const char* (*item_name)(void* ctx)
        const_struct_AVOption *option
        int version
        int log_level_offset_offset
        int parent_log_context_offset
        void* (*child_next)(void *obj, void *prev)
        const_struct_AVClass* (*child_class_next)(const_struct_AVClass *prev)
        AVClassCategory category
        AVClassCategory (*get_category)(void* ctx)
        int (*query_ranges)(AVOptionRanges **, void *obj, const char *key, int flags)
#rational.h
cdef extern from "../libs/ffmpeg/include/libavutil/rational.h":
    struct AVRational:
        int num                    #< numerator
        int den                    #< denominator 
#util/frame.h
cdef extern from "../libs/ffmpeg/include/libavutil/frame.h":
    enum AVFrameSideDataType:
        AV_FRAME_DATA_PANSCAN,
        AV_FRAME_DATA_A53_CC,
        AV_FRAME_DATA_STEREO3D,
        AV_FRAME_DATA_MATRIXENCODING,
        AV_FRAME_DATA_DOWNMIX_INFO,
        AV_FRAME_DATA_REPLAYGAIN,
        AV_FRAME_DATA_DISPLAYMATRIX,
        AV_FRAME_DATA_AFD,
        AV_FRAME_DATA_MOTION_VECTORS,
        AV_FRAME_DATA_SKIP_SAMPLES,
        AV_FRAME_DATA_AUDIO_SERVICE_TYPE,
        AV_FRAME_DATA_MASTERING_DISPLAY_METADATA,
        AV_FRAME_DATA_GOP_TIMECODE,
        AV_FRAME_DATA_SPHERICAL,
        AV_FRAME_DATA_CONTENT_LIGHT_LEVEL,
        AV_FRAME_DATA_ICC_PROFILE,
        AV_FRAME_DATA_QP_TABLE_PROPERTIES,
        AV_FRAME_DATA_QP_TABLE_DATA

    enum:
        AV_NUM_DATA_POINTERS = 8
        AV_FRAME_FLAG_CORRUPT= (1 << 0)
        AV_FRAME_FLAG_DISCARD= (1 << 2)
        FF_DECODE_ERROR_INVALID_BITSTREAM=   1
        FF_DECODE_ERROR_MISSING_REFERENCE=   2
    struct AVFrameSideData:
        AVFrameSideDataType type
        uint8_t *data
        int size
        AVDictionary *metadata
        AVBufferRef *buf

    struct AVFrame:
        uint8_t *data[AV_NUM_DATA_POINTERS]
        int linesize[AV_NUM_DATA_POINTERS]
        uint8_t **extended_data
        int width, height
        int nb_samples
        int format
        int key_frame
        AVPictureType pict_type
        AVRational sample_aspect_ratio
        int64_t pts
		#if FF_API_PKT_PTS
        int64_t pkt_pts
		#endif
        int64_t pkt_dts
        int coded_picture_number
        int display_picture_number
        int quality
        void *opaque
		#if FF_API_ERROR_FRAME
        uint64_t error[AV_NUM_DATA_POINTERS]
		#endif
        int repeat_pict
        int interlaced_frame
        int top_field_first
        int palette_has_changed
        int64_t reordered_opaque
        int sample_rate
        uint64_t channel_layout
        AVBufferRef *buf[AV_NUM_DATA_POINTERS]
        AVBufferRef **extended_buf
        int                nb_extended_buf
        AVFrameSideData **side_data
        int                        nb_side_data
        int flags
        AVColorRange color_range
        AVColorPrimaries color_primaries
        AVColorTransferCharacteristic color_trc
        AVColorSpace colorspace
        AVChromaLocation chroma_location
        int64_t best_effort_timestamp
        int64_t pkt_pos
        int64_t pkt_duration
        AVDictionary *metadata
        int decode_error_flags
        int channels
        int pkt_size

#if FF_API_FRAME_QP
        int8_t *qscale_table
        int qstride
        int qscale_type
        AVBufferRef *qp_table_buf
#endif
        AVBufferRef *hw_frames_ctx
        AVBufferRef *opaque_ref
        size_t crop_top
        size_t crop_bottom
        size_t crop_left
        size_t crop_right
        AVBufferRef *private_ref

#libavcodec.h
cdef extern from "../libs/ffmpeg/include/libavcodec/avcodec.h":
    ctypedef struct AVCodecDefault
    ctypedef struct AVCodecInternal
    enum AVAudioServiceType:
        AV_AUDIO_SERVICE_TYPE_MAIN              = 0
        AV_AUDIO_SERVICE_TYPE_EFFECTS           = 1
        AV_AUDIO_SERVICE_TYPE_VISUALLY_IMPAIRED = 2
        AV_AUDIO_SERVICE_TYPE_HEARING_IMPAIRED  = 3
        AV_AUDIO_SERVICE_TYPE_DIALOGUE          = 4
        AV_AUDIO_SERVICE_TYPE_COMMENTARY        = 5
        AV_AUDIO_SERVICE_TYPE_EMERGENCY         = 6
        AV_AUDIO_SERVICE_TYPE_VOICE_OVER        = 7
        AV_AUDIO_SERVICE_TYPE_KARAOKE           = 8
        AV_AUDIO_SERVICE_TYPE_NB
    enum AVFieldOrder:
        AV_FIELD_UNKNOWN,
        AV_FIELD_PROGRESSIVE,
        AV_FIELD_TT,          #< Top coded_first, top displayed first
        AV_FIELD_BB,          #< Bottom coded first, bottom displayed first
        AV_FIELD_TB,          #< Top coded first, bottom displayed first
        AV_FIELD_BT          #< Bottom coded first, top displayed first
    
    enum AVDiscard:
        AVDISCARD_NONE   = -16 # discard nothing
        AVDISCARD_DEFAULT=   0 # discard useless packets like 0 size packets in avi
        AVDISCARD_NONREF =   8 # discard all non reference
        AVDISCARD_BIDIR  =  16 # discard all bidirectional frames
        AVDISCARD_NONINTRA= 24 # discard all non intra frames
        AVDISCARD_NONKEY =  32 # discard all frames except keyframes
        AVDISCARD_ALL    =  48 # discard all
    enum AVCodecID:
        AV_CODEC_ID_NONE,
        AV_CODEC_ID_MPEG1VIDEO,
        AV_CODEC_ID_MPEG2VIDEO, #< preferred ID for MPEG-1/2 video decoding
        AV_CODEC_ID_H261,
        AV_CODEC_ID_H263,
        AV_CODEC_ID_RV10,
        AV_CODEC_ID_RV20,
        AV_CODEC_ID_MJPEG,
        AV_CODEC_ID_MJPEGB,
        AV_CODEC_ID_LJPEG,
        AV_CODEC_ID_SP5X,
        AV_CODEC_ID_JPEGLS,
        AV_CODEC_ID_MPEG4,
        AV_CODEC_ID_RAWVIDEO,
        AV_CODEC_ID_MSMPEG4V1,
        AV_CODEC_ID_MSMPEG4V2,
        AV_CODEC_ID_MSMPEG4V3,
        AV_CODEC_ID_WMV1,
        AV_CODEC_ID_WMV2,
        AV_CODEC_ID_H263P,
        AV_CODEC_ID_H263I,
        AV_CODEC_ID_FLV1,
        AV_CODEC_ID_SVQ1,
        AV_CODEC_ID_SVQ3,
        AV_CODEC_ID_DVVIDEO,
        AV_CODEC_ID_HUFFYUV,
        AV_CODEC_ID_CYUV,
        AV_CODEC_ID_H264,
        AV_CODEC_ID_INDEO3,
        AV_CODEC_ID_VP3,
        AV_CODEC_ID_THEORA,
        AV_CODEC_ID_ASV1,
        AV_CODEC_ID_ASV2,
        AV_CODEC_ID_FFV1,
        AV_CODEC_ID_4XM,
        AV_CODEC_ID_VCR1,
        AV_CODEC_ID_CLJR,
        AV_CODEC_ID_MDEC,
        AV_CODEC_ID_ROQ,
        AV_CODEC_ID_INTERPLAY_VIDEO,
        AV_CODEC_ID_XAN_WC3,
        AV_CODEC_ID_XAN_WC4,
        AV_CODEC_ID_RPZA,
        AV_CODEC_ID_CINEPAK,
        AV_CODEC_ID_WS_VQA,
        AV_CODEC_ID_MSRLE,
        AV_CODEC_ID_MSVIDEO1,
        AV_CODEC_ID_IDCIN,
        AV_CODEC_ID_8BPS,
        AV_CODEC_ID_SMC,
        AV_CODEC_ID_FLIC,
        AV_CODEC_ID_TRUEMOTION1,
        AV_CODEC_ID_VMDVIDEO,
        AV_CODEC_ID_MSZH,
        AV_CODEC_ID_ZLIB,
        AV_CODEC_ID_QTRLE,
        AV_CODEC_ID_TSCC,
        AV_CODEC_ID_ULTI,
        AV_CODEC_ID_QDRAW,
        AV_CODEC_ID_VIXL,
        AV_CODEC_ID_QPEG,
        AV_CODEC_ID_PNG,
        AV_CODEC_ID_PPM,
        AV_CODEC_ID_PBM,
        AV_CODEC_ID_PGM,
        AV_CODEC_ID_PGMYUV,
        AV_CODEC_ID_PAM,
        AV_CODEC_ID_FFVHUFF,
        AV_CODEC_ID_RV30,
        AV_CODEC_ID_RV40,
        AV_CODEC_ID_VC1,
        AV_CODEC_ID_WMV3,
        AV_CODEC_ID_LOCO,
        AV_CODEC_ID_WNV1,
        AV_CODEC_ID_AASC,
        AV_CODEC_ID_INDEO2,
        AV_CODEC_ID_FRAPS,
        AV_CODEC_ID_TRUEMOTION2,
        AV_CODEC_ID_BMP,
        AV_CODEC_ID_CSCD,
        AV_CODEC_ID_MMVIDEO,
        AV_CODEC_ID_ZMBV,
        AV_CODEC_ID_AVS,
        AV_CODEC_ID_SMACKVIDEO,
        AV_CODEC_ID_NUV,
        AV_CODEC_ID_KMVC,
        AV_CODEC_ID_FLASHSV,
        AV_CODEC_ID_CAVS,
        AV_CODEC_ID_JPEG2000,
        AV_CODEC_ID_VMNC,
        AV_CODEC_ID_VP5,
        AV_CODEC_ID_VP6,
        AV_CODEC_ID_VP6F,
        AV_CODEC_ID_TARGA,
        AV_CODEC_ID_DSICINVIDEO,
        AV_CODEC_ID_TIERTEXSEQVIDEO,
        AV_CODEC_ID_TIFF,
        AV_CODEC_ID_GIF,
        AV_CODEC_ID_DXA,
        AV_CODEC_ID_DNXHD,
        AV_CODEC_ID_THP,
        AV_CODEC_ID_SGI,
        AV_CODEC_ID_C93,
        AV_CODEC_ID_BETHSOFTVID,
        AV_CODEC_ID_PTX,
        AV_CODEC_ID_TXD,
        AV_CODEC_ID_VP6A,
        AV_CODEC_ID_AMV,
        AV_CODEC_ID_VB,
        AV_CODEC_ID_PCX,
        AV_CODEC_ID_SUNRAST,
        AV_CODEC_ID_INDEO4,
        AV_CODEC_ID_INDEO5,
        AV_CODEC_ID_MIMIC,
        AV_CODEC_ID_RL2,
        AV_CODEC_ID_ESCAPE124,
        AV_CODEC_ID_DIRAC,
        AV_CODEC_ID_BFI,
        AV_CODEC_ID_CMV,
        AV_CODEC_ID_MOTIONPIXELS,
        AV_CODEC_ID_TGV,
        AV_CODEC_ID_TGQ,
        AV_CODEC_ID_TQI,
        AV_CODEC_ID_AURA,
        AV_CODEC_ID_AURA2,
        AV_CODEC_ID_V210X,
        AV_CODEC_ID_TMV,
        AV_CODEC_ID_V210,
        AV_CODEC_ID_DPX,
        AV_CODEC_ID_MAD,
        AV_CODEC_ID_FRWU,
        AV_CODEC_ID_FLASHSV2,
        AV_CODEC_ID_CDGRAPHICS,
        AV_CODEC_ID_R210,
        AV_CODEC_ID_ANM,
        AV_CODEC_ID_BINKVIDEO,
        AV_CODEC_ID_IFF_ILBM,

        AV_CODEC_ID_KGV1,
        AV_CODEC_ID_YOP,
        AV_CODEC_ID_VP8,
        AV_CODEC_ID_PICTOR,
        AV_CODEC_ID_ANSI,
        AV_CODEC_ID_A64_MULTI,
        AV_CODEC_ID_A64_MULTI5,
        AV_CODEC_ID_R10K,
        AV_CODEC_ID_MXPEG,
        AV_CODEC_ID_LAGARITH,
        AV_CODEC_ID_PRORES,
        AV_CODEC_ID_JV,
        AV_CODEC_ID_DFA,
        AV_CODEC_ID_WMV3IMAGE,
        AV_CODEC_ID_VC1IMAGE,
        AV_CODEC_ID_UTVIDEO,
        AV_CODEC_ID_BMV_VIDEO,
        AV_CODEC_ID_VBLE,
        AV_CODEC_ID_DXTORY,
        AV_CODEC_ID_V410,
        AV_CODEC_ID_XWD,
        AV_CODEC_ID_CDXL,
        AV_CODEC_ID_XBM,
        AV_CODEC_ID_ZEROCODEC,
        AV_CODEC_ID_MSS1,
        AV_CODEC_ID_MSA1,
        AV_CODEC_ID_TSCC2,
        AV_CODEC_ID_MTS2,
        AV_CODEC_ID_CLLC,
        AV_CODEC_ID_MSS2,
        AV_CODEC_ID_VP9,
        AV_CODEC_ID_AIC,
        AV_CODEC_ID_ESCAPE130,
        AV_CODEC_ID_G2M,
        AV_CODEC_ID_WEBP,
        AV_CODEC_ID_HNM4_VIDEO,
        AV_CODEC_ID_HEVC,

        AV_CODEC_ID_FIC,
        AV_CODEC_ID_ALIAS_PIX,
        AV_CODEC_ID_BRENDER_PIX,
        AV_CODEC_ID_PAF_VIDEO,
        AV_CODEC_ID_EXR,
        AV_CODEC_ID_VP7,
        AV_CODEC_ID_SANM,
        AV_CODEC_ID_SGIRLE,
        AV_CODEC_ID_MVC1,
        AV_CODEC_ID_MVC2,
        AV_CODEC_ID_HQX,
        AV_CODEC_ID_TDSC,
        AV_CODEC_ID_HQ_HQA,
        AV_CODEC_ID_HAP,
        AV_CODEC_ID_DDS,
        AV_CODEC_ID_DXV,
        AV_CODEC_ID_SCREENPRESSO,
        AV_CODEC_ID_RSCC,
        AV_CODEC_ID_AVS2,

        AV_CODEC_ID_Y41P = 0x8000,
        AV_CODEC_ID_AVRP,
        AV_CODEC_ID_012V,
        AV_CODEC_ID_AVUI,
        AV_CODEC_ID_AYUV,
        AV_CODEC_ID_TARGA_Y216,
        AV_CODEC_ID_V308,
        AV_CODEC_ID_V408,
        AV_CODEC_ID_YUV4,
        AV_CODEC_ID_AVRN,
        AV_CODEC_ID_CPIA,
        AV_CODEC_ID_XFACE,
        AV_CODEC_ID_SNOW,
        AV_CODEC_ID_SMVJPEG,
        AV_CODEC_ID_APNG,
        AV_CODEC_ID_DAALA,
        AV_CODEC_ID_CFHD,
        AV_CODEC_ID_TRUEMOTION2RT,
        AV_CODEC_ID_M101,
        AV_CODEC_ID_MAGICYUV,
        AV_CODEC_ID_SHEERVIDEO,
        AV_CODEC_ID_YLC,
        AV_CODEC_ID_PSD,
        AV_CODEC_ID_PIXLET,
        AV_CODEC_ID_SPEEDHQ,
        AV_CODEC_ID_FMVC,
        AV_CODEC_ID_SCPR,
        AV_CODEC_ID_CLEARVIDEO,
        AV_CODEC_ID_XPM,
        AV_CODEC_ID_AV1,
        AV_CODEC_ID_BITPACKED,
        AV_CODEC_ID_MSCC,
        AV_CODEC_ID_SRGC,
        AV_CODEC_ID_SVG,
        AV_CODEC_ID_GDV,
        AV_CODEC_ID_FITS,
        AV_CODEC_ID_IMM4,
        AV_CODEC_ID_PROSUMER,
        AV_CODEC_ID_MWSC,
        AV_CODEC_ID_WCMV,


        AV_CODEC_ID_FIRST_AUDIO = 0x10000,         #< A dummy id pointing at the start of audio codecs
        AV_CODEC_ID_PCM_S16LE = 0x10000,
        AV_CODEC_ID_PCM_S16BE,
        AV_CODEC_ID_PCM_U16LE,
        AV_CODEC_ID_PCM_U16BE,
        AV_CODEC_ID_PCM_S8,
        AV_CODEC_ID_PCM_U8,
        AV_CODEC_ID_PCM_MULAW,
        AV_CODEC_ID_PCM_ALAW,
        AV_CODEC_ID_PCM_S32LE,
        AV_CODEC_ID_PCM_S32BE,
        AV_CODEC_ID_PCM_U32LE,
        AV_CODEC_ID_PCM_U32BE,
        AV_CODEC_ID_PCM_S24LE,
        AV_CODEC_ID_PCM_S24BE,
        AV_CODEC_ID_PCM_U24LE,
        AV_CODEC_ID_PCM_U24BE,
        AV_CODEC_ID_PCM_S24DAUD,
        AV_CODEC_ID_PCM_ZORK,
        AV_CODEC_ID_PCM_S16LE_PLANAR,
        AV_CODEC_ID_PCM_DVD,
        AV_CODEC_ID_PCM_F32BE,
        AV_CODEC_ID_PCM_F32LE,
        AV_CODEC_ID_PCM_F64BE,
        AV_CODEC_ID_PCM_F64LE,
        AV_CODEC_ID_PCM_BLURAY,
        AV_CODEC_ID_PCM_LXF,
        AV_CODEC_ID_S302M,
        AV_CODEC_ID_PCM_S8_PLANAR,
        AV_CODEC_ID_PCM_S24LE_PLANAR,
        AV_CODEC_ID_PCM_S32LE_PLANAR,
        AV_CODEC_ID_PCM_S16BE_PLANAR,

        AV_CODEC_ID_PCM_S64LE = 0x10800,
        AV_CODEC_ID_PCM_S64BE,
        AV_CODEC_ID_PCM_F16LE,
        AV_CODEC_ID_PCM_F24LE,


        AV_CODEC_ID_ADPCM_IMA_QT = 0x11000,
        AV_CODEC_ID_ADPCM_IMA_WAV,
        AV_CODEC_ID_ADPCM_IMA_DK3,
        AV_CODEC_ID_ADPCM_IMA_DK4,
        AV_CODEC_ID_ADPCM_IMA_WS,
        AV_CODEC_ID_ADPCM_IMA_SMJPEG,
        AV_CODEC_ID_ADPCM_MS,
        AV_CODEC_ID_ADPCM_4XM,
        AV_CODEC_ID_ADPCM_XA,
        AV_CODEC_ID_ADPCM_ADX,
        AV_CODEC_ID_ADPCM_EA,
        AV_CODEC_ID_ADPCM_G726,
        AV_CODEC_ID_ADPCM_CT,
        AV_CODEC_ID_ADPCM_SWF,
        AV_CODEC_ID_ADPCM_YAMAHA,
        AV_CODEC_ID_ADPCM_SBPRO_4,
        AV_CODEC_ID_ADPCM_SBPRO_3,
        AV_CODEC_ID_ADPCM_SBPRO_2,
        AV_CODEC_ID_ADPCM_THP,
        AV_CODEC_ID_ADPCM_IMA_AMV,
        AV_CODEC_ID_ADPCM_EA_R1,
        AV_CODEC_ID_ADPCM_EA_R3,
        AV_CODEC_ID_ADPCM_EA_R2,
        AV_CODEC_ID_ADPCM_IMA_EA_SEAD,
        AV_CODEC_ID_ADPCM_IMA_EA_EACS,
        AV_CODEC_ID_ADPCM_EA_XAS,
        AV_CODEC_ID_ADPCM_EA_MAXIS_XA,
        AV_CODEC_ID_ADPCM_IMA_ISS,
        AV_CODEC_ID_ADPCM_G722,
        AV_CODEC_ID_ADPCM_IMA_APC,
        AV_CODEC_ID_ADPCM_VIMA,

        AV_CODEC_ID_ADPCM_AFC = 0x11800,
        AV_CODEC_ID_ADPCM_IMA_OKI,
        AV_CODEC_ID_ADPCM_DTK,
        AV_CODEC_ID_ADPCM_IMA_RAD,
        AV_CODEC_ID_ADPCM_G726LE,
        AV_CODEC_ID_ADPCM_THP_LE,
        AV_CODEC_ID_ADPCM_PSX,
        AV_CODEC_ID_ADPCM_AICA,
        AV_CODEC_ID_ADPCM_IMA_DAT4,
        AV_CODEC_ID_ADPCM_MTAF,

        # AMR */
        AV_CODEC_ID_AMR_NB = 0x12000,
        AV_CODEC_ID_AMR_WB,

        # RealAudio codecs*/
        AV_CODEC_ID_RA_144 = 0x13000,
        AV_CODEC_ID_RA_288,

        # various DPCM codecs */
        AV_CODEC_ID_ROQ_DPCM = 0x14000,
        AV_CODEC_ID_INTERPLAY_DPCM,
        AV_CODEC_ID_XAN_DPCM,
        AV_CODEC_ID_SOL_DPCM,

        AV_CODEC_ID_SDX2_DPCM = 0x14800,
        AV_CODEC_ID_GREMLIN_DPCM,

        # audio codecs */
        AV_CODEC_ID_MP2 = 0x15000,
        AV_CODEC_ID_MP3, #< preferred ID for decoding MPEG audio layer 1, 2 or 3
        AV_CODEC_ID_AAC,
        AV_CODEC_ID_AC3,
        AV_CODEC_ID_DTS,
        AV_CODEC_ID_VORBIS,
        AV_CODEC_ID_DVAUDIO,
        AV_CODEC_ID_WMAV1,
        AV_CODEC_ID_WMAV2,
        AV_CODEC_ID_MACE3,
        AV_CODEC_ID_MACE6,
        AV_CODEC_ID_VMDAUDIO,
        AV_CODEC_ID_FLAC,
        AV_CODEC_ID_MP3ADU,
        AV_CODEC_ID_MP3ON4,
        AV_CODEC_ID_SHORTEN,
        AV_CODEC_ID_ALAC,
        AV_CODEC_ID_WESTWOOD_SND1,
        AV_CODEC_ID_GSM, #< as in Berlin toast format
        AV_CODEC_ID_QDM2,
        AV_CODEC_ID_COOK,
        AV_CODEC_ID_TRUESPEECH,
        AV_CODEC_ID_TTA,
        AV_CODEC_ID_SMACKAUDIO,
        AV_CODEC_ID_QCELP,
        AV_CODEC_ID_WAVPACK,
        AV_CODEC_ID_DSICINAUDIO,
        AV_CODEC_ID_IMC,
        AV_CODEC_ID_MUSEPACK7,
        AV_CODEC_ID_MLP,
        AV_CODEC_ID_GSM_MS, # as found in WAV */
        AV_CODEC_ID_ATRAC3,
        AV_CODEC_ID_APE,
        AV_CODEC_ID_NELLYMOSER,
        AV_CODEC_ID_MUSEPACK8,
        AV_CODEC_ID_SPEEX,
        AV_CODEC_ID_WMAVOICE,
        AV_CODEC_ID_WMAPRO,
        AV_CODEC_ID_WMALOSSLESS,
        AV_CODEC_ID_ATRAC3P,
        AV_CODEC_ID_EAC3,
        AV_CODEC_ID_SIPR,
        AV_CODEC_ID_MP1,
        AV_CODEC_ID_TWINVQ,
        AV_CODEC_ID_TRUEHD,
        AV_CODEC_ID_MP4ALS,
        AV_CODEC_ID_ATRAC1,
        AV_CODEC_ID_BINKAUDIO_RDFT,
        AV_CODEC_ID_BINKAUDIO_DCT,
        AV_CODEC_ID_AAC_LATM,
        AV_CODEC_ID_QDMC,
        AV_CODEC_ID_CELT,
        AV_CODEC_ID_G723_1,
        AV_CODEC_ID_G729,
        AV_CODEC_ID_8SVX_EXP,
        AV_CODEC_ID_8SVX_FIB,
        AV_CODEC_ID_BMV_AUDIO,
        AV_CODEC_ID_RALF,
        AV_CODEC_ID_IAC,
        AV_CODEC_ID_ILBC,
        AV_CODEC_ID_OPUS,
        AV_CODEC_ID_COMFORT_NOISE,
        AV_CODEC_ID_TAK,
        AV_CODEC_ID_METASOUND,
        AV_CODEC_ID_PAF_AUDIO,
        AV_CODEC_ID_ON2AVC,
        AV_CODEC_ID_DSS_SP,
        AV_CODEC_ID_CODEC2,

        AV_CODEC_ID_FFWAVESYNTH = 0x15800,
        AV_CODEC_ID_SONIC,
        AV_CODEC_ID_SONIC_LS,
        AV_CODEC_ID_EVRC,
        AV_CODEC_ID_SMV,
        AV_CODEC_ID_DSD_LSBF,
        AV_CODEC_ID_DSD_MSBF,
        AV_CODEC_ID_DSD_LSBF_PLANAR,
        AV_CODEC_ID_DSD_MSBF_PLANAR,
        AV_CODEC_ID_4GV,
        AV_CODEC_ID_INTERPLAY_ACM,
        AV_CODEC_ID_XMA1,
        AV_CODEC_ID_XMA2,
        AV_CODEC_ID_DST,
        AV_CODEC_ID_ATRAC3AL,
        AV_CODEC_ID_ATRAC3PAL,
        AV_CODEC_ID_DOLBY_E,
        AV_CODEC_ID_APTX,
        AV_CODEC_ID_APTX_HD,
        AV_CODEC_ID_SBC,
        AV_CODEC_ID_ATRAC9,

        # subtitle codecs */
        AV_CODEC_ID_FIRST_SUBTITLE = 0x17000,                  #< A dummy ID pointing at the start of subtitle codecs.
        AV_CODEC_ID_DVD_SUBTITLE = 0x17000,
        AV_CODEC_ID_DVB_SUBTITLE,
        AV_CODEC_ID_TEXT,  #< raw UTF-8 text
        AV_CODEC_ID_XSUB,
        AV_CODEC_ID_SSA,
        AV_CODEC_ID_MOV_TEXT,
        AV_CODEC_ID_HDMV_PGS_SUBTITLE,
        AV_CODEC_ID_DVB_TELETEXT,
        AV_CODEC_ID_SRT,

        AV_CODEC_ID_MICRODVD   = 0x17800,
        AV_CODEC_ID_EIA_608,
        AV_CODEC_ID_JACOSUB,
        AV_CODEC_ID_SAMI,
        AV_CODEC_ID_REALTEXT,
        AV_CODEC_ID_STL,
        AV_CODEC_ID_SUBVIEWER1,
        AV_CODEC_ID_SUBVIEWER,
        AV_CODEC_ID_SUBRIP,
        AV_CODEC_ID_WEBVTT,
        AV_CODEC_ID_MPL2,
        AV_CODEC_ID_VPLAYER,
        AV_CODEC_ID_PJS,
        AV_CODEC_ID_ASS,
        AV_CODEC_ID_HDMV_TEXT_SUBTITLE,
        AV_CODEC_ID_TTML,

        # other specific kind of codecs (generally used for attachments) */
        AV_CODEC_ID_FIRST_UNKNOWN = 0x18000,                   #< A dummy ID pointing at the start of various fake codecs.
        AV_CODEC_ID_TTF = 0x18000,

        AV_CODEC_ID_SCTE_35, #< Contain timestamp estimated through PCR of program stream.
        AV_CODEC_ID_BINTEXT        = 0x18800,
        AV_CODEC_ID_XBIN,
        AV_CODEC_ID_IDF,
        AV_CODEC_ID_OTF,
        AV_CODEC_ID_SMPTE_KLV,
        AV_CODEC_ID_DVD_NAV,
        AV_CODEC_ID_TIMED_ID3,
        AV_CODEC_ID_BIN_DATA,


        AV_CODEC_ID_PROBE = 0x19000, #< codec_id is not known (like AV_CODEC_ID_NONE) but lavf should attempt to identify it

        AV_CODEC_ID_MPEG2TS = 0x20000, #*< _FAKE_ codec to indicate a raw MPEG-2 TS
                                                               
        AV_CODEC_ID_MPEG4SYSTEMS = 0x20001, #*< _FAKE_ codec to indicate a MPEG-4 Systems
                                                               
        AV_CODEC_ID_FFMETADATA = 0x21000,   #< Dummy codec for streams containing only metadata information.
        AV_CODEC_ID_WRAPPED_AVFRAME = 0x21001, #< Passthrough codec, AVFrames wrapped in AVPacket
        
    enum AVPacketSideDataType:
        AV_PKT_DATA_PALETTE,
        AV_PKT_DATA_NEW_EXTRADATA,
        AV_PKT_DATA_PARAM_CHANGE,
        AV_PKT_DATA_H263_MB_INFO,
        AV_PKT_DATA_REPLAYGAIN,
        AV_PKT_DATA_DISPLAYMATRIX,
        AV_PKT_DATA_STEREO3D,
        AV_PKT_DATA_AUDIO_SERVICE_TYPE,
        AV_PKT_DATA_QUALITY_STATS,
        AV_PKT_DATA_FALLBACK_TRACK,
        AV_PKT_DATA_CPB_PROPERTIES,
        AV_PKT_DATA_SKIP_SAMPLES,
        AV_PKT_DATA_JP_DUALMONO,
        AV_PKT_DATA_STRINGS_METADATA,
        AV_PKT_DATA_SUBTITLE_POSITION,
        AV_PKT_DATA_MATROSKA_BLOCKADDITIONAL,
        AV_PKT_DATA_WEBVTT_IDENTIFIER,
        AV_PKT_DATA_WEBVTT_SETTINGS,
        AV_PKT_DATA_METADATA_UPDATE,
        AV_PKT_DATA_MPEGTS_STREAM_ID,
        AV_PKT_DATA_MASTERING_DISPLAY_METADATA,
        AV_PKT_DATA_SPHERICAL,
        AV_PKT_DATA_CONTENT_LIGHT_LEVEL,
        AV_PKT_DATA_A53_CC,
        AV_PKT_DATA_ENCRYPTION_INIT_INFO,
        AV_PKT_DATA_ENCRYPTION_INFO,
        AV_PKT_DATA_NB

    enum:
        FF_COMPRESSION_DEFAULT= -1,
        FF_PRED_LEFT=   0,
        FF_PRED_PLANE=  1,
        FF_PRED_MEDIAN =2,
        FF_CMP_SAD=                  0,
        FF_CMP_SSE =                 1,
        FF_CMP_SATD=                 2,
        FF_CMP_DCT =                 3,
        FF_CMP_PSNR =                4,
        FF_CMP_BIT  =                5,
        FF_CMP_RD   =                6,
        FF_CMP_ZERO =                7,
        FF_CMP_VSAD  =           8,
        FF_CMP_VSSE  =           9,
        FF_CMP_NSSE  =           10,
        FF_CMP_W53  =                11,
        FF_CMP_W97  =                12,
        FF_CMP_DCTMAX =          13,
        FF_CMP_DCT264=           14,
        FF_CMP_MEDIAN_SAD =  15,
        FF_CMP_CHROMA =          256,
        SLICE_FLAG_CODED_ORDER=        0x0001, #/< draw_horiz_band() is called in coded order instead of display
        SLICE_FLAG_ALLOW_FIELD=        0x0002, #/< allow draw_horiz_band() with field slices (MPEG-2 field pics)
        SLICE_FLAG_ALLOW_PLANE=        0x0004, #/< allow draw_horiz_band() with 1 component at a time (SVQ1)
        FF_MB_DECISION_SIMPLE= 0 ,           #/< uses mb_cmp
        FF_MB_DECISION_BITS =  1,                #/< chooses the one which needs the fewest bits
        FF_MB_DECISION_RD =        2 ,           #/< rate distortion
        FF_CODER_TYPE_VLC =          0,
        FF_CODER_TYPE_AC =           1,
        FF_CODER_TYPE_RAW =          2,
        FF_CODER_TYPE_RLE =          3,
        FF_BUG_AUTODETECT=           1 , #/< autodetection
        FF_BUG_XVID_ILACE =          4,
        FF_BUG_UMP4        =                 8,
        FF_BUG_NO_PADDING=           16,
        FF_BUG_AMV =                         32,
        FF_BUG_QPEL_CHROMA  =        64,
        FF_BUG_STD_QPEL =                128,
        FF_BUG_QPEL_CHROMA2 =        256,
        FF_BUG_DIRECT_BLOCKSIZE= 512,
        FF_BUG_EDGE =                        1024,
        FF_BUG_HPEL_CHROMA =         2048,
        FF_BUG_DC_CLIP  =                4096,
        FF_BUG_MS   =                        8192, #/< Work around various bugs in Microsoft's broken decoders.
        FF_BUG_TRUNCATED =          16384,
        FF_BUG_IEDGE =                  32768,
        FF_COMPLIANCE_VERY_STRICT=   2, #/< Strictly conform to an older more strict version of the spec or reference software.
        FF_COMPLIANCE_STRICT =           1, #/< Strictly conform to all the things in the spec no matter what consequences.
        FF_COMPLIANCE_NORMAL=                0,
        FF_COMPLIANCE_UNOFFICIAL=   -1, #/< Allow unofficial extensions
        FF_COMPLIANCE_EXPERIMENTAL= -2, #/< Allow nonstandardized experimental things.
        FF_EC_GUESS_MVS=   1,
        FF_EC_DEBLOCK=         2,
        FF_EC_FAVOR_INTER= 256,
        FF_PROFILE_UNKNOWN= -99,
        FF_PROFILE_RESERVED= -100,
        FF_PROFILE_AAC_MAIN= 0,
        FF_PROFILE_AAC_LOW=  1,
        FF_PROFILE_AAC_SSR=  2,
        FF_PROFILE_AAC_LTP=  3,
        FF_PROFILE_AAC_HE=   4,
        FF_PROFILE_AAC_HE_V2= 28,
        FF_PROFILE_AAC_LD=   22,
        FF_PROFILE_AAC_ELD=  38,
        FF_PROFILE_MPEG2_AAC_LOW= 128,
        FF_PROFILE_MPEG2_AAC_HE=  131,
        FF_PROFILE_DNXHD=                 0,
        FF_PROFILE_DNXHR_LB=          1,
        FF_PROFILE_DNXHR_SQ=          2,
        FF_PROFILE_DNXHR_HQ=          3,
        FF_PROFILE_DNXHR_HQX=         4,
        FF_PROFILE_DNXHR_444=         5,
        FF_PROFILE_DTS=                 20,
        FF_PROFILE_DTS_ES=          30,
        FF_PROFILE_DTS_96_24=   40,
        FF_PROFILE_DTS_HD_HRA=  50,
        FF_PROFILE_DTS_HD_MA=   60,
        FF_PROFILE_DTS_EXPRESS= 70,
        FF_PROFILE_MPEG2_422=        0,
        FF_PROFILE_MPEG2_HIGH=   1,
        FF_PROFILE_MPEG2_SS=         2,
        FF_PROFILE_MPEG2_SNR_SCALABLE=  3,
        FF_PROFILE_MPEG2_MAIN=   4,
        FF_PROFILE_MPEG2_SIMPLE= 5,
        FF_PROFILE_H264_CONSTRAINED=  (1<<9) , # 8+1 constraint_set1_flag
        FF_PROFILE_H264_INTRA=                (1<<11), # 8+3 constraint_set3_flag
        FF_PROFILE_H264_BASELINE =                        66,
        FF_PROFILE_H264_CONSTRAINED_BASELINE= (66|FF_PROFILE_H264_CONSTRAINED),
        FF_PROFILE_H264_MAIN=                                 77,
        FF_PROFILE_H264_EXTENDED =                        88,
        FF_PROFILE_H264_HIGH  =                           100,
        FF_PROFILE_H264_HIGH_10 =                         110,
        FF_PROFILE_H264_HIGH_10_INTRA=                (110|FF_PROFILE_H264_INTRA),
        FF_PROFILE_H264_MULTIVIEW_HIGH=           118,
        FF_PROFILE_H264_HIGH_422 =                        122,
        FF_PROFILE_H264_HIGH_422_INTRA=           (122|FF_PROFILE_H264_INTRA),
        FF_PROFILE_H264_STEREO_HIGH =                 128,
        FF_PROFILE_H264_HIGH_444   =                  144,
        FF_PROFILE_H264_HIGH_444_PREDICTIVE=  244,
        FF_PROFILE_H264_HIGH_444_INTRA=           (244|FF_PROFILE_H264_INTRA),
        FF_PROFILE_H264_CAVLC_444 =                   44,
        FF_PROFILE_VC1_SIMPLE=   0,
        FF_PROFILE_VC1_MAIN =        1,
        FF_PROFILE_VC1_COMPLEX = 2,
        FF_PROFILE_VC1_ADVANCED= 3,
        FF_PROFILE_MPEG4_SIMPLE=                                         0,
        FF_PROFILE_MPEG4_SIMPLE_SCALABLE=                        1,
        FF_PROFILE_MPEG4_CORE=                                           2,
        FF_PROFILE_MPEG4_MAIN =                                          3,
        FF_PROFILE_MPEG4_N_BIT =                                         4,
        FF_PROFILE_MPEG4_SCALABLE_TEXTURE=                   5,
        FF_PROFILE_MPEG4_SIMPLE_FACE_ANIMATION =         6,
        FF_PROFILE_MPEG4_BASIC_ANIMATED_TEXTURE=         7,
        FF_PROFILE_MPEG4_HYBRID =                                        8,
        FF_PROFILE_MPEG4_ADVANCED_REAL_TIME=                 9,
        FF_PROFILE_MPEG4_CORE_SCALABLE =                        10,
        FF_PROFILE_MPEG4_ADVANCED_CODING =                  11,
        FF_PROFILE_MPEG4_ADVANCED_CORE =                        12,
        FF_PROFILE_MPEG4_ADVANCED_SCALABLE_TEXTURE =13,
        FF_PROFILE_MPEG4_SIMPLE_STUDIO =                        14,
        FF_PROFILE_MPEG4_ADVANCED_SIMPLE =                  15,
        FF_PROFILE_JPEG2000_CSTREAM_RESTRICTION_0 =  1,
        FF_PROFILE_JPEG2000_CSTREAM_RESTRICTION_1=   2,
        FF_PROFILE_JPEG2000_CSTREAM_NO_RESTRICTION=  32768,
        FF_PROFILE_JPEG2000_DCINEMA_2K =                         3,
        FF_PROFILE_JPEG2000_DCINEMA_4K =                         4,
        FF_PROFILE_VP9_0=                                                        0,
        FF_PROFILE_VP9_1=                                                        1,
        FF_PROFILE_VP9_2 =                                                   2,
        FF_PROFILE_VP9_3 =                                                   3,
        FF_PROFILE_HEVC_MAIN  =                                          1,
        FF_PROFILE_HEVC_MAIN_10  =                                   2,
        FF_PROFILE_HEVC_MAIN_STILL_PICTURE=                  3,
        FF_PROFILE_HEVC_REXT   =                                         4,
        FF_PROFILE_AV1_MAIN =                                                0,
        FF_PROFILE_AV1_HIGH  =                                           1,
        FF_PROFILE_AV1_PROFESSIONAL   =                          2,
        FF_PROFILE_MJPEG_HUFFMAN_BASELINE_DCT =                   0xc0,
        FF_PROFILE_MJPEG_HUFFMAN_EXTENDED_SEQUENTIAL_DCT= 0xc1,
        FF_PROFILE_MJPEG_HUFFMAN_PROGRESSIVE_DCT=                 0xc2,
        FF_PROFILE_MJPEG_HUFFMAN_LOSSLESS =                           0xc3,
        FF_PROFILE_MJPEG_JPEG_LS =                                                0xf7,
        FF_PROFILE_SBC_MSBC=                                                 1,
        FF_LEVEL_UNKNOWN =-99,
        FF_DEBUG_PICT_INFO=   1,
        FF_DEBUG_RC           =   2,
        FF_DEBUG_BITSTREAM =  4,
        FF_DEBUG_MB_TYPE  =   8,
        FF_DEBUG_QP                =  16,
        FF_DEBUG_MV                =  32,
        FF_DEBUG_DCT_COEFF =  0x00000040,
        FF_DEBUG_SKIP          =  0x00000080,
        FF_DEBUG_STARTCODE =  0x00000100,
        FF_DEBUG_ER                =  0x00000400,
        FF_DEBUG_MMCO          =  0x00000800,
        FF_DEBUG_BUGS          =  0x00001000,
        FF_DEBUG_VIS_QP        =  0x00002000,
        FF_DEBUG_VIS_MB_TYPE =0x00004000,
        FF_DEBUG_BUFFERS        = 0x00008000,
        FF_DEBUG_THREADS        = 0x00010000,
        FF_DEBUG_GREEN_MD  =  0x00800000,
        FF_DEBUG_NOMC           = 0x01000000,
        FF_DEBUG_VIS_MV_P_FOR = 0x00000001, # visualize forward predicted MVs of P-frames
        FF_DEBUG_VIS_MV_B_FOR = 0x00000002, # visualize forward predicted MVs of B-frames
        FF_DEBUG_VIS_MV_B_BACK= 0x00000004, # visualize backward predicted MVs of B-frames
        AV_EF_CRCCHECK = (1<<0),
        AV_EF_BITSTREAM= (1<<1) ,                 #/< detect bitstream specification deviations
        AV_EF_BUFFER   = (1<<2) ,                 #/< detect improper bitstream length
        AV_EF_EXPLODE  = (1<<3) ,                 #/< abort decoding on minor error detection
        AV_EF_IGNORE_ERR =(1<<15) ,           #/< ignore errors and continue
        AV_EF_CAREFUL  =  (1<<16) ,           #/< consider things that violate the spec, are fast to calculate and have not been seen in the wild as errors
        AV_EF_COMPLIANT  =(1<<17) ,           #/< consider all spec non compliances as errors
        AV_EF_AGGRESSIVE =(1<<18) ,           #/< consider things that a sane encoder should not do as an error
        FF_DCT_AUTO  =  0,
        FF_DCT_FASTINT= 1,
        FF_DCT_INT   =  2,
        FF_DCT_MMX        = 3,
        FF_DCT_ALTIVEC= 5,
        FF_DCT_FAAN        =6,
        FF_IDCT_AUTO         =         0,
        FF_IDCT_INT           =        1,
        FF_IDCT_SIMPLE        =        2,
        FF_IDCT_SIMPLEMMX =        3,
        FF_IDCT_ARM           =        7,
        FF_IDCT_ALTIVEC   =        8,
        FF_IDCT_SIMPLEARM  =   10,
        FF_IDCT_XVID           =   14,
        FF_IDCT_SIMPLEARMV5TE= 16,
        FF_IDCT_SIMPLEARMV6=   17,
        FF_IDCT_FAAN          =        20,
        FF_IDCT_SIMPLENEON =   22,
        FF_IDCT_NONE                =  24 ,  #/* Used by XvMC to extract IDCT coefficients with FF_IDCT_PERM_NONE */
        FF_IDCT_SIMPLEAUTO =   128,
        FF_THREAD_FRAME  = 1, #/< Decode more than one frame at once
        FF_THREAD_SLICE =  2, #/< Decode more than one part of a single frame at once
        FF_SUB_CHARENC_MODE_DO_NOTHING = -1,  #/< do nothing (demuxer outputs a stream supposed to be already in UTF-8, or the codec is bitmap for instance)
        FF_SUB_CHARENC_MODE_AUTOMATIC =   0,  #/< libavcodec will select the mode itself
        FF_SUB_CHARENC_MODE_PRE_DECODER=  1 , #/< the AVPacket data needs to be recoded to UTF-8 before being fed to the decoder, requires iconv
        FF_SUB_CHARENC_MODE_IGNORE         =  2 , #/< neither convert the subtitles, nor check them for valid UTF-8
        FF_DEBUG_VIS_MV_P_FOR=  0x00000001, #visualize forward predicted MVs of P frames
        FF_DEBUG_VIS_MV_B_FOR = 0x00000002, #visualize forward predicted MVs of B frames
        FF_DEBUG_VIS_MV_B_BACK= 0x00000004, #visualize backward predicted MVs of B frames
        FF_CODEC_PROPERTY_LOSSLESS   =         0x00000001,
        FF_CODEC_PROPERTY_CLOSED_CAPTIONS= 0x00000002,
        FF_SUB_TEXT_FMT_ASS   =                   0,
        FF_SUB_TEXT_FMT_ASS_WITH_TIMINGS =1

    struct AVCodecContext:
        const AVClass *av_class
        int log_level_offset
        AVMediaType codec_type   #/* see AVMEDIA_TYPE_xxx */
        const_struct_AVCodec  *codec
        AVCodecID  codec_id   # /* see AV_CODEC_ID_xxx */
        unsigned int codec_tag
        void *priv_data
        AVCodecInternal *internal
        void *opaque
        int64_t bit_rate
        int bit_rate_tolerance
        int global_quality
        int compression_level
        int flags
        int flags2
        uint8_t *extradata
        int extradata_size
        AVRational time_base
        int ticks_per_frame
        int delay
        int width, height
        int coded_width, coded_height
        int gop_size
        AVPixelFormat pix_fmt
        void (*draw_horiz_band)(AVCodecContext *s,const AVFrame *src, int offset[AV_NUM_DATA_POINTERS],int y, int type, int height)
        AVPixelFormat (*get_format)(AVCodecContext *s, const AVPixelFormat * fmt)
        int max_b_frames
        float b_quant_factor
#if FF_API_PRIVATE_OPT
        int b_frame_strategy
#endif
        float b_quant_offset
        int has_b_frames
#if FF_API_PRIVATE_OPT
        int mpeg_quant
#endif
        float i_quant_factor
        float i_quant_offset
        float lumi_masking
        float temporal_cplx_masking
        float spatial_cplx_masking
        float p_masking
        float dark_masking
        int slice_count
#if FF_API_PRIVATE_OPT
        int prediction_method
#endif
        int *slice_offset
        AVRational sample_aspect_ratio
        int me_cmp
        int me_sub_cmp
        int mb_cmp
        int ildct_cmp
        int dia_size
        int last_predictor_count
#if FF_API_PRIVATE_OPT
        int pre_me
#endif
        int me_pre_cmp
        int pre_dia_size
        int me_subpel_quality
        int me_range
        int slice_flags
        int mb_decision
        uint16_t *intra_matrix
        uint16_t *inter_matrix
#if FF_API_PRIVATE_OPT
        int scenechange_threshold
        int noise_reduction
#endif
        int intra_dc_precision
        int skip_top
        int skip_bottom
        int mb_lmin
        int mb_lmax
#if FF_API_PRIVATE_OPT
        int me_penalty_compensation
#endif
        int bidir_refine
#if FF_API_PRIVATE_OPT
        int brd_scale
#endif
        int keyint_min
        int refs

#if FF_API_PRIVATE_OPT
        int chromaoffset
#endif
        int mv0_threshold
#if FF_API_PRIVATE_OPT
        int b_sensitivity
#endif
        AVColorPrimaries color_primaries
        AVColorTransferCharacteristic color_trc
        AVColorSpace colorspace
        AVColorRange color_range
        AVChromaLocation chroma_sample_location
        int slices
        AVFieldOrder field_order
        int sample_rate #/< samples per second
        int channels        #/< number of audio channels
        AVSampleFormat sample_fmt  #/< sample format
        int frame_size
        int frame_number
        int block_align
        int cutoff
        uint64_t channel_layout
        uint64_t request_channel_layout
        AVAudioServiceType audio_service_type
        AVSampleFormat request_sample_fmt
        int (*get_buffer2)(AVCodecContext *s, AVFrame *frame, int flags)
        int refcounted_frames
        float qcompress  #/< amount of qscale change between easy & hard scenes (0.0-1.0)
        float qblur          #/< amount of qscale smoothing over time (0.0-1.0)
        int qmin
        int qmax
        int max_qdiff
        int rc_buffer_size
        int rc_override_count
        RcOverride *rc_override
        int64_t rc_max_rate
        int64_t rc_min_rate
        float rc_max_available_vbv_use
        float rc_min_vbv_overflow_use
        int rc_initial_buffer_occupancy
#if FF_API_CODER_TYPE
        int coder_type
#endif /* FF_API_CODER_TYPE */
#if FF_API_PRIVATE_OPT
        int context_model
#endif
#if FF_API_PRIVATE_OPT
        int frame_skip_threshold
        int frame_skip_factor
        int frame_skip_exp
        int frame_skip_cmp
#endif /* FF_API_PRIVATE_OPT */
        int trellis
#if FF_API_PRIVATE_OPT
        int min_prediction_order
        int max_prediction_order
        int64_t timecode_frame_start
#endif
#if FF_API_RTP_CALLBACK
        void (*rtp_callback)(AVCodecContext *avctx, void *data, int size, int mb_nb)
#endif
#if FF_API_PRIVATE_OPT
        int rtp_payload_size   #/* The size of the RTP payload: the coder will  */
#endif
#if FF_API_STAT_BITS
        int mv_bits
        int header_bits
        int i_tex_bits
        int p_tex_bits
        int i_count
        int p_count
        int skip_count
        int misc_bits
        int frame_bits
#endif
        char *stats_out
        char *stats_in
        int workaround_bugs
        int strict_std_compliance
        int error_concealment
        int debug
        int err_recognition
        int64_t reordered_opaque
        const_struct_AVHWAccel *hwaccel
        void *hwaccel_context
        uint64_t error[AV_NUM_DATA_POINTERS]
        int dct_algo
        int idct_algo
        int bits_per_coded_sample
        int bits_per_raw_sample
#if FF_API_CODED_FRAME
        AVFrame *coded_frame
#endif
        int thread_count
        int thread_type
        int active_thread_type
        int thread_safe_callbacks
        int (*execute)( AVCodecContext *c, int (*func)( AVCodecContext *c2, void *arg), void *arg2, int *ret, int count, int size)
        int (*execute2)( AVCodecContext *c, int (*func)( AVCodecContext *c2, void *arg, int jobnr, int threadnr), void *arg2, int *ret, int count)
        int nsse_weight
        int profile
        int level
        AVDiscard skip_loop_filter
        AVDiscard skip_idct
        AVDiscard skip_frame
        uint8_t *subtitle_header
        int subtitle_header_size
#if FF_API_VBV_DELAY
        uint64_t vbv_delay
#endif
#if FF_API_SIDEDATA_ONLY_PKT
        int side_data_only_packets
#endif
        int initial_padding
        AVRational framerate
        AVPixelFormat sw_pix_fmt
        AVRational pkt_timebase
        const AVCodecDescriptor *codec_descriptor
#if !FF_API_LOWRES
        int lowres
#endif
        int64_t pts_correction_num_faulty_pts #/ Number of incorrect PTS values so far
        int64_t pts_correction_num_faulty_dts #/ Number of incorrect DTS values so far
        int64_t pts_correction_last_pts           #/ PTS of the last frame
        int64_t pts_correction_last_dts           #/ DTS of the last frame
        char *sub_charenc
        int sub_charenc_mode
        int skip_alpha
        int seek_preroll
#if !FF_API_DEBUG_MV
        int debug_mv
#endif
        uint16_t *chroma_intra_matrix
        uint8_t *dump_separator
        char *codec_whitelist
        unsigned properties
        AVPacketSideData *coded_side_data
        int                        nb_coded_side_data
        AVBufferRef *hw_frames_ctx
        int sub_text_format
        int trailing_padding
        int64_t max_pixels
        AVBufferRef *hw_device_ctx
        int hwaccel_flags
        int apply_cropping
        int extra_hw_frames

    struct AVCodecDescriptor:
        AVCodecID       id
        AVMediaType     type
        const char      *name           # Name of the codec described by this descriptor, non-empty and unique for each descriptor
        const char      *long_name      # A more descriptive name for this codec. May be NULL
        int             props           # Codec properties, a combination of AV_CODEC_PROP_* flags
        const char      **mime_types    # MIME type(s) associated with the codec
        const_struct_AVProfile *profiles
    struct RcOverride:
        int start_frame
        int end_frame
        int qscale # If this is 0 then quality_factor will be used instead
        float quality_factor

    struct AVPacketSideData:
        uint8_t *data
        int      size
        AVPacketSideDataType type

    struct AVCodecParameters:
        AVMediaType codec_type
        AVCodecID   codec_id
        uint32_t                 codec_tag
        uint8_t *extradata
        int          extradata_size
        int format
        int64_t bit_rate
        int bits_per_coded_sample
        int bits_per_raw_sample
        int profile
        int level
        int width
        int height
        AVRational sample_aspect_ratio
        AVFieldOrder  field_order
        AVColorRange color_range
        AVColorPrimaries color_primaries
        AVColorTransferCharacteristic color_trc
        AVColorSpace  color_space
        AVChromaLocation chroma_location
        int video_delay
        uint64_t channel_layout
        int          channels
        int          sample_rate
        int          block_align
        int          frame_size
        int initial_padding
        int trailing_padding
        int seek_preroll

    struct AVProfile:
        int         profile
        char *      name                    #< short name for the profile

    struct AVCodec:
        char *        name
        char *        long_name
        AVMediaType   type
        AVCodecID     id
        int           capabilities    # see CODEC_CAP_*
        AVRational *supported_framerates #< array of supported framerates, or NULL if any, array is terminated by {0,0}
        AVPixelFormat *pix_fmts      #< array of supported pixel formats, or NULL if unknown, array is terminated by -1
        int *supported_samplerates   #< array of supported audio samplerates, or NULL if unknown, array is terminated by 0
        AVSampleFormat *sample_fmts  #< array of supported sample formats, or NULL if unknown, array is terminated by -1
        uint64_t *channel_layouts    #< array of support channel layouts, or NULL if unknown. array is terminated by 0
#if FF_API_LOWRES
        uint8_t max_lowres       #< maximum value for lowres supported by the decoder, no direct access, use av_codec_get_max_lowres()
#endif
        const AVClass *priv_class      #< AVClass for the private context
        AVProfile *profiles      #< array of recognized profiles, or NULL if unknown, array is terminated by {FF_PROFILE_UNKNOWN}
        const char *wrapper_name
        int priv_data_size
        AVCodec *next

        int (*init_thread_copy)(AVCodecContext *)
        int (*update_thread_context)(AVCodecContext *dst, const AVCodecContext *src)
        AVCodecDefault *defaults
        void (*init_static_data)(AVCodec *codec)
        int (*init)(AVCodecContext *)
        int (*encode_sub)(AVCodecContext *, uint8_t *buf, int buf_size, const_struct_AVSubtitle *sub)
        int (*encode2)(AVCodecContext *avctx, AVPacket *avpkt, const AVFrame *frame, int *got_packet_ptr)
        int (*decode)(AVCodecContext *, void *outdata, int *outdata_size, AVPacket *avpkt)
        int (*close)(AVCodecContext *)
        int (*send_frame)(AVCodecContext *avctx, const AVFrame *frame)
        int (*receive_packet)(AVCodecContext *avctx, AVPacket *avpkt)
        int (*receive_frame)(AVCodecContext *avctx, AVFrame *frame)
        void (*flush)(AVCodecContext *)
        int caps_internal
        const char *bsfs
        #AVCodecHWConfigInternal **hw_configs
    struct AVPacket:
        AVBufferRef *buf
        int64_t pts
        int64_t dts
        uint8_t *data
        int   size
        int   stream_index
        int   flags
        AVPacketSideData *side_data
        int side_data_elems
        int64_t duration
        int64_t pos                                                        #///< byte position in stream, -1 if unknown
        #if FF_API_CONVERGENCE_DURATION
        int64_t convergence_duration

      
#util/avutil.h
cdef extern from "../libs/ffmpeg/include/libavutil/avutil.h":
    enum AVPictureType:
        AV_PICTURE_TYPE_NONE = 0, #/< Undefined
        AV_PICTURE_TYPE_I,         #/< Intra
        AV_PICTURE_TYPE_P,         #/< Predicted
        AV_PICTURE_TYPE_B,         #/< Bi-dir predicted
        AV_PICTURE_TYPE_S,         #/< S(GMC)-VOP MPEG-4
        AV_PICTURE_TYPE_SI,        #/< Switching Intra
        AV_PICTURE_TYPE_SP,        #/< Switching Predicted
        AV_PICTURE_TYPE_BI      #/< BI type

    enum AVMediaType:
        AVMEDIA_TYPE_UNKNOWN = -1,
        AVMEDIA_TYPE_VIDEO,
        AVMEDIA_TYPE_AUDIO,
        AVMEDIA_TYPE_DATA,
        AVMEDIA_TYPE_SUBTITLE,
        AVMEDIA_TYPE_ATTACHMENT,
        AVMEDIA_TYPE_NB
    enum:        
        AV_NOPTS_VALUE = <int64_t>0x8000000000000000
        AV_TIME_BASE = 1000000


#util/buffer.h
cdef extern from "../libs/ffmpeg/include/libavutil/buffer.h":
    ctypedef struct AVBuffer
    struct AVBufferRef:
        AVBuffer *buffer
        uint8_t *data       #< The data buffer
        int size            #< Size of data in bytes


#util/samplefmt.h
cdef extern from "../libs/ffmpeg/include/libavutil/samplefmt.h":
    enum AVSampleFormat:
        AV_SAMPLE_FMT_NONE = -1,
        AV_SAMPLE_FMT_U8,                  #/< unsigned 8 bits
        AV_SAMPLE_FMT_S16,                 #/< signed 16 bits
        AV_SAMPLE_FMT_S32,                 #/< signed 32 bits
        AV_SAMPLE_FMT_FLT,                 #/< float
        AV_SAMPLE_FMT_DBL,                 #/< double

        AV_SAMPLE_FMT_U8P,                 #/< unsigned 8 bits, planar
        AV_SAMPLE_FMT_S16P,                #/< signed 16 bits, planar
        AV_SAMPLE_FMT_S32P,                #/< signed 32 bits, planar
        AV_SAMPLE_FMT_FLTP,                #/< float, planar
        AV_SAMPLE_FMT_DBLP,                #/< double, planar
        AV_SAMPLE_FMT_S64,                 #/< signed 64 bits
        AV_SAMPLE_FMT_S64P,                #/< signed 64 bits, planar

        AV_SAMPLE_FMT_NB                   #/< Number of sample formats. DO NOT USE if linking dynamically
#util/pixfmt.h
cdef extern from "../libs/ffmpeg/include/libavutil/pixfmt.h":
    enum AVColorPrimaries:
        AVCOL_PRI_RESERVED0   = 0,
        AVCOL_PRI_BT709           = 1,  #/< also ITU-R BT1361 / IEC 61966-2-4 / SMPTE RP177 Annex B
        AVCOL_PRI_UNSPECIFIED = 2,
        AVCOL_PRI_RESERVED        = 3,
        AVCOL_PRI_BT470M          = 4,  #/< also FCC Title 47 Code of Federal Regulations 73.682 (a)(20)

        AVCOL_PRI_BT470BG         = 5,  #/< also ITU-R BT601-6 625 / ITU-R BT1358 625 / ITU-R BT1700 625 PAL & SECAM
        AVCOL_PRI_SMPTE170M   = 6,  #/< also ITU-R BT601-6 525 / ITU-R BT1358 525 / ITU-R BT1700 NTSC
        AVCOL_PRI_SMPTE240M   = 7,  #/< functionally identical to above
        AVCOL_PRI_FILM                = 8,  #/< colour filters using Illuminant C
        AVCOL_PRI_BT2020          = 9,  #/< ITU-R BT2020
        AVCOL_PRI_SMPTE428        = 10, #/< SMPTE ST 428-1 (CIE 1931 XYZ)
        AVCOL_PRI_SMPTEST428_1 = AVCOL_PRI_SMPTE428,
        AVCOL_PRI_SMPTE431        = 11, #/< SMPTE ST 431-2 (2011) / DCI P3
        AVCOL_PRI_SMPTE432        = 12, #/< SMPTE ST 432-1 (2010) / P3 D65 / Display P3
        AVCOL_PRI_JEDEC_P22   = 22, #/< JEDEC P22 phosphors
        AVCOL_PRI_NB                                #/< Not part of ABI
    enum AVColorRange:
        AVCOL_RANGE_UNSPECIFIED = 0,
        AVCOL_RANGE_MPEG                = 1, #/< the normal 219*2^(n-8) "MPEG" YUV ranges
        AVCOL_RANGE_JPEG                = 2, #/< the normal         2^n-1   "JPEG" YUV ranges
        AVCOL_RANGE_NB                           #/< Not part of ABI
    enum AVColorTransferCharacteristic:
        AVCOL_TRC_RESERVED0        = 0,
        AVCOL_TRC_BT709                = 1,  #/< also ITU-R BT1361
        AVCOL_TRC_UNSPECIFIED  = 2,
        AVCOL_TRC_RESERVED         = 3,
        AVCOL_TRC_GAMMA22          = 4,  #/< also ITU-R BT470M / ITU-R BT1700 625 PAL & SECAM
        AVCOL_TRC_GAMMA28          = 5,  #/< also ITU-R BT470BG
        AVCOL_TRC_SMPTE170M        = 6,  #/< also ITU-R BT601-6 525 or 625 / ITU-R BT1358 525 or 625 / ITU-R BT1700 NTSC
        AVCOL_TRC_SMPTE240M        = 7,
        AVCOL_TRC_LINEAR           = 8,  #/< "Linear transfer characteristics"
        AVCOL_TRC_LOG                  = 9,  #/< "Logarithmic transfer characteristic (100:1 range)"
        AVCOL_TRC_LOG_SQRT         = 10, #/< "Logarithmic transfer characteristic (100 * Sqrt(10) : 1 range)"
        AVCOL_TRC_IEC61966_2_4 = 11, #/< IEC 61966-2-4
        AVCOL_TRC_BT1361_ECG   = 12, #/< ITU-R BT1361 Extended Colour Gamut
        AVCOL_TRC_IEC61966_2_1 = 13, #/< IEC 61966-2-1 (sRGB or sYCC)
        AVCOL_TRC_BT2020_10        = 14, #/< ITU-R BT2020 for 10-bit system
        AVCOL_TRC_BT2020_12        = 15, #/< ITU-R BT2020 for 12-bit system
        AVCOL_TRC_SMPTE2084        = 16, #/< SMPTE ST 2084 for 10-, 12-, 14- and 16-bit systems
        AVCOL_TRC_SMPTEST2084  = AVCOL_TRC_SMPTE2084,
        AVCOL_TRC_SMPTE428         = 17, #/< SMPTE ST 428-1
        AVCOL_TRC_SMPTEST428_1 = AVCOL_TRC_SMPTE428,
        AVCOL_TRC_ARIB_STD_B67 = 18, #/< ARIB STD-B67, known as "Hybrid log-gamma"
        AVCOL_TRC_NB                                 #/< Not part of ABI

    enum AVColorSpace:
        AVCOL_SPC_RGB                 = 0,  #/< order of coefficients is actually GBR, also IEC 61966-2-1 (sRGB)
        AVCOL_SPC_BT709           = 1,  #/< also ITU-R BT1361 / IEC 61966-2-4 xvYCC709 / SMPTE RP177 Annex B
        AVCOL_SPC_UNSPECIFIED = 2,
        AVCOL_SPC_RESERVED        = 3,
        AVCOL_SPC_FCC                 = 4,  #/< FCC Title 47 Code of Federal Regulations 73.682 (a)(20)
        AVCOL_SPC_BT470BG         = 5,  #/< also ITU-R BT601-6 625 / ITU-R BT1358 625 / ITU-R BT1700 625 PAL & SECAM / IEC 61966-2-4 xvYCC601
        AVCOL_SPC_SMPTE170M   = 6,  #/< also ITU-R BT601-6 525 / ITU-R BT1358 525 / ITU-R BT1700 NTSC
        AVCOL_SPC_SMPTE240M   = 7,  #/< functionally identical to above
        AVCOL_SPC_YCGCO           = 8,  #/< Used by Dirac / VC-2 and H.264 FRext, see ITU-T SG16
        AVCOL_SPC_YCOCG           = AVCOL_SPC_YCGCO,
        AVCOL_SPC_BT2020_NCL  = 9,  #/< ITU-R BT2020 non-constant luminance system
        AVCOL_SPC_BT2020_CL   = 10, #/< ITU-R BT2020 constant luminance system
        AVCOL_SPC_SMPTE2085   = 11, #/< SMPTE 2085, Y'D'zD'x
        AVCOL_SPC_CHROMA_DERIVED_NCL = 12, #/< Chromaticity-derived non-constant luminance system
        AVCOL_SPC_CHROMA_DERIVED_CL = 13, #/< Chromaticity-derived constant luminance system
        AVCOL_SPC_ICTCP           = 14, #/< ITU-R BT.2100-0, ICtCp
        AVCOL_SPC_NB                                #/< Not part of ABI

    enum AVChromaLocation:
        AVCHROMA_LOC_UNSPECIFIED = 0,
        AVCHROMA_LOC_LEFT                = 1, #///< MPEG-2/4 4:2:0, H.264 default for 4:2:0
        AVCHROMA_LOC_CENTER          = 2, #///< MPEG-1 4:2:0, JPEG 4:2:0, H.263 4:2:0
        AVCHROMA_LOC_TOPLEFT         = 3, #///< ITU-R 601, SMPTE 274M 296M S314M(DV 4:1:1), mpeg2 4:2:2
        AVCHROMA_LOC_TOP                 = 4,
        AVCHROMA_LOC_BOTTOMLEFT  = 5,
        AVCHROMA_LOC_BOTTOM          = 6,
        AVCHROMA_LOC_NB                           #///< Not part of ABI
    enum AVPixelFormat:
        AV_PIX_FMT_NONE = -1,
        AV_PIX_FMT_YUV420P,   #//< planar YUV 4:2:0, 12bpp, (1 Cr & Cb sample per 2x2 Y samples)
        AV_PIX_FMT_YUYV422,   #//< packed YUV 4:2:2, 16bpp, Y0 Cb Y1 Cr
        AV_PIX_FMT_RGB24,         #//< packed RGB 8:8:8, 24bpp, RGBRGB...
        AV_PIX_FMT_BGR24,         #//< packed RGB 8:8:8, 24bpp, BGRBGR...
        AV_PIX_FMT_YUV422P,   #//< planar YUV 4:2:2, 16bpp, (1 Cr & Cb sample per 2x1 Y samples)
        AV_PIX_FMT_YUV444P,   #//< planar YUV 4:4:4, 24bpp, (1 Cr & Cb sample per 1x1 Y samples)
        AV_PIX_FMT_YUV410P,   #//< planar YUV 4:1:0,  9bpp, (1 Cr & Cb sample per 4x4 Y samples)
        AV_PIX_FMT_YUV411P,   #//< planar YUV 4:1:1, 12bpp, (1 Cr & Cb sample per 4x1 Y samples)
        AV_PIX_FMT_GRAY8,         #//<                Y                ,  8bpp
        AV_PIX_FMT_MONOWHITE, #//<                Y                ,  1bpp, 0 is white, 1 is black, in each byte pixels are ordered from the msb to the lsb
        AV_PIX_FMT_MONOBLACK, #//<                Y                ,  1bpp, 0 is black, 1 is white, in each byte pixels are ordered from the msb to the lsb
        AV_PIX_FMT_PAL8,          #//< 8 bits with AV_PIX_FMT_RGB32 palette
        AV_PIX_FMT_YUVJ420P,  #//< planar YUV 4:2:0, 12bpp, full scale (JPEG), deprecated in favor of AV_PIX_FMT_YUV420P and setting color_range
        AV_PIX_FMT_YUVJ422P,  #//< planar YUV 4:2:2, 16bpp, full scale (JPEG), deprecated in favor of AV_PIX_FMT_YUV422P and setting color_range
        AV_PIX_FMT_YUVJ444P,  #//< planar YUV 4:4:4, 24bpp, full scale (JPEG), deprecated in favor of AV_PIX_FMT_YUV444P and setting color_range
        AV_PIX_FMT_UYVY422,   #//< packed YUV 4:2:2, 16bpp, Cb Y0 Cr Y1
        AV_PIX_FMT_UYYVYY411, #//< packed YUV 4:1:1, 12bpp, Cb Y0 Y1 Cr Y2 Y3
        AV_PIX_FMT_BGR8,          #//< packed RGB 3:3:2,  8bpp, (msb)2B 3G 3R(lsb)
        AV_PIX_FMT_BGR4,          #//< packed RGB 1:2:1 bitstream,  4bpp, (msb)1B 2G 1R(lsb), a byte contains two pixels, the first pixel in the byte is the one composed by the 4 msb bits
        AV_PIX_FMT_BGR4_BYTE, #//< packed RGB 1:2:1,  8bpp, (msb)1B 2G 1R(lsb)
        AV_PIX_FMT_RGB8,          #//< packed RGB 3:3:2,  8bpp, (msb)2R 3G 3B(lsb)
        AV_PIX_FMT_RGB4,          #//< packed RGB 1:2:1 bitstream,  4bpp, (msb)1R 2G 1B(lsb), a byte contains two pixels, the first pixel in the byte is the one composed by the 4 msb bits
        AV_PIX_FMT_RGB4_BYTE, #//< packed RGB 1:2:1,  8bpp, (msb)1R 2G 1B(lsb)
        AV_PIX_FMT_NV12,          #//< planar YUV 4:2:0, 12bpp, 1 plane for Y and 1 plane for the UV components, which are interleaved (first byte U and the following byte V)
        AV_PIX_FMT_NV21,          #//< as above, but U and V bytes are swapped

        AV_PIX_FMT_ARGB,          #//< packed ARGB 8:8:8:8, 32bpp, ARGBARGB...
        AV_PIX_FMT_RGBA,          #//< packed RGBA 8:8:8:8, 32bpp, RGBARGBA...
        AV_PIX_FMT_ABGR,          #//< packed ABGR 8:8:8:8, 32bpp, ABGRABGR...
        AV_PIX_FMT_BGRA,          #//< packed BGRA 8:8:8:8, 32bpp, BGRABGRA...

        AV_PIX_FMT_GRAY16BE,  #//<                Y                , 16bpp, big-endian
        AV_PIX_FMT_GRAY16LE,  #//<                Y                , 16bpp, little-endian
        AV_PIX_FMT_YUV440P,   #//< planar YUV 4:4:0 (1 Cr & Cb sample per 1x2 Y samples)
        AV_PIX_FMT_YUVJ440P,  #//< planar YUV 4:4:0 full scale (JPEG), deprecated in favor of AV_PIX_FMT_YUV440P and setting color_range
        AV_PIX_FMT_YUVA420P,  #//< planar YUV 4:2:0, 20bpp, (1 Cr & Cb sample per 2x2 Y & A samples)
        AV_PIX_FMT_RGB48BE,   #//< packed RGB 16:16:16, 48bpp, 16R, 16G, 16B, the 2-byte value for each R/G/B component is stored as big-endian
        AV_PIX_FMT_RGB48LE,   #//< packed RGB 16:16:16, 48bpp, 16R, 16G, 16B, the 2-byte value for each R/G/B component is stored as little-endian

        AV_PIX_FMT_RGB565BE,  #//< packed RGB 5:6:5, 16bpp, (msb)   5R 6G 5B(lsb), big-endian
        AV_PIX_FMT_RGB565LE,  #//< packed RGB 5:6:5, 16bpp, (msb)   5R 6G 5B(lsb), little-endian
        AV_PIX_FMT_RGB555BE,  #//< packed RGB 5:5:5, 16bpp, (msb)1X 5R 5G 5B(lsb), big-endian   , X=unused/undefined
        AV_PIX_FMT_RGB555LE,  #//< packed RGB 5:5:5, 16bpp, (msb)1X 5R 5G 5B(lsb), little-endian, X=unused/undefined

        AV_PIX_FMT_BGR565BE,  #//< packed BGR 5:6:5, 16bpp, (msb)   5B 6G 5R(lsb), big-endian
        AV_PIX_FMT_BGR565LE,  #//< packed BGR 5:6:5, 16bpp, (msb)   5B 6G 5R(lsb), little-endian
        AV_PIX_FMT_BGR555BE,  #//< packed BGR 5:5:5, 16bpp, (msb)1X 5B 5G 5R(lsb), big-endian   , X=unused/undefined
        AV_PIX_FMT_BGR555LE,  #//< packed BGR 5:5:5, 16bpp, (msb)1X 5B 5G 5R(lsb), little-endian, X=unused/undefined

		#if FF_API_VAAPI
        AV_PIX_FMT_VAAPI_MOCO, #//< HW acceleration through VA API at motion compensation entry-point, Picture.data[3] contains a vaapi_render_state struct which contains macroblocks as well as various fields extracted from headers
        AV_PIX_FMT_VAAPI_IDCT, #//< HW acceleration through VA API at IDCT entry-point, Picture.data[3] contains a vaapi_render_state struct which contains fields extracted from headers
        AV_PIX_FMT_VAAPI_VLD,  #//< HW decoding through VA API, Picture.data[3] contains a VASurfaceID
        AV_PIX_FMT_VAAPI = AV_PIX_FMT_VAAPI_VLD,
		#else
        #AV_PIX_FMT_VAAPI,
		#endif

        AV_PIX_FMT_YUV420P16LE,  #//< planar YUV 4:2:0, 24bpp, (1 Cr & Cb sample per 2x2 Y samples), little-endian
        AV_PIX_FMT_YUV420P16BE,  #//< planar YUV 4:2:0, 24bpp, (1 Cr & Cb sample per 2x2 Y samples), big-endian
        AV_PIX_FMT_YUV422P16LE,  #//< planar YUV 4:2:2, 32bpp, (1 Cr & Cb sample per 2x1 Y samples), little-endian
        AV_PIX_FMT_YUV422P16BE,  #//< planar YUV 4:2:2, 32bpp, (1 Cr & Cb sample per 2x1 Y samples), big-endian
        AV_PIX_FMT_YUV444P16LE,  #//< planar YUV 4:4:4, 48bpp, (1 Cr & Cb sample per 1x1 Y samples), little-endian
        AV_PIX_FMT_YUV444P16BE,  #//< planar YUV 4:4:4, 48bpp, (1 Cr & Cb sample per 1x1 Y samples), big-endian
        AV_PIX_FMT_DXVA2_VLD,        #//< HW decoding through DXVA2, Picture.data[3] contains a LPDIRECT3DSURFACE9 pointer

        AV_PIX_FMT_RGB444LE,  #//< packed RGB 4:4:4, 16bpp, (msb)4X 4R 4G 4B(lsb), little-endian, X=unused/undefined
        AV_PIX_FMT_RGB444BE,  #//< packed RGB 4:4:4, 16bpp, (msb)4X 4R 4G 4B(lsb), big-endian,        X=unused/undefined
        AV_PIX_FMT_BGR444LE,  #//< packed BGR 4:4:4, 16bpp, (msb)4X 4B 4G 4R(lsb), little-endian, X=unused/undefined
        AV_PIX_FMT_BGR444BE,  #//< packed BGR 4:4:4, 16bpp, (msb)4X 4B 4G 4R(lsb), big-endian,        X=unused/undefined
        AV_PIX_FMT_YA8,           #//< 8 bits gray, 8 bits alpha

        AV_PIX_FMT_Y400A = AV_PIX_FMT_YA8, #//< alias for AV_PIX_FMT_YA8
        AV_PIX_FMT_GRAY8A= AV_PIX_FMT_YA8, #//< alias for AV_PIX_FMT_YA8

        AV_PIX_FMT_BGR48BE,   #//< packed RGB 16:16:16, 48bpp, 16B, 16G, 16R, the 2-byte value for each R/G/B component is stored as big-endian
        AV_PIX_FMT_BGR48LE,   #//< packed RGB 16:16:16, 48bpp, 16B, 16G, 16R, the 2-byte value for each R/G/B component is stored as little-endian

        AV_PIX_FMT_YUV420P9BE, #//< planar YUV 4:2:0, 13.5bpp, (1 Cr & Cb sample per 2x2 Y samples), big-endian
        AV_PIX_FMT_YUV420P9LE, #//< planar YUV 4:2:0, 13.5bpp, (1 Cr & Cb sample per 2x2 Y samples), little-endian
        AV_PIX_FMT_YUV420P10BE,#//< planar YUV 4:2:0, 15bpp, (1 Cr & Cb sample per 2x2 Y samples), big-endian
        AV_PIX_FMT_YUV420P10LE,#//< planar YUV 4:2:0, 15bpp, (1 Cr & Cb sample per 2x2 Y samples), little-endian
        AV_PIX_FMT_YUV422P10BE,#//< planar YUV 4:2:2, 20bpp, (1 Cr & Cb sample per 2x1 Y samples), big-endian
        AV_PIX_FMT_YUV422P10LE,#//< planar YUV 4:2:2, 20bpp, (1 Cr & Cb sample per 2x1 Y samples), little-endian
        AV_PIX_FMT_YUV444P9BE, #//< planar YUV 4:4:4, 27bpp, (1 Cr & Cb sample per 1x1 Y samples), big-endian
        AV_PIX_FMT_YUV444P9LE, #//< planar YUV 4:4:4, 27bpp, (1 Cr & Cb sample per 1x1 Y samples), little-endian
        AV_PIX_FMT_YUV444P10BE,#//< planar YUV 4:4:4, 30bpp, (1 Cr & Cb sample per 1x1 Y samples), big-endian
        AV_PIX_FMT_YUV444P10LE,#//< planar YUV 4:4:4, 30bpp, (1 Cr & Cb sample per 1x1 Y samples), little-endian
        AV_PIX_FMT_YUV422P9BE, #//< planar YUV 4:2:2, 18bpp, (1 Cr & Cb sample per 2x1 Y samples), big-endian
        AV_PIX_FMT_YUV422P9LE, #//< planar YUV 4:2:2, 18bpp, (1 Cr & Cb sample per 2x1 Y samples), little-endian
        AV_PIX_FMT_GBRP,          #//< planar GBR 4:4:4 24bpp
        AV_PIX_FMT_GBR24P = AV_PIX_FMT_GBRP,  #// alias for #AV_PIX_FMT_GBRP
        AV_PIX_FMT_GBRP9BE,   #//< planar GBR 4:4:4 27bpp, big-endian
        AV_PIX_FMT_GBRP9LE,   #//< planar GBR 4:4:4 27bpp, little-endian
        AV_PIX_FMT_GBRP10BE,  #//< planar GBR 4:4:4 30bpp, big-endian
        AV_PIX_FMT_GBRP10LE,  #//< planar GBR 4:4:4 30bpp, little-endian
        AV_PIX_FMT_GBRP16BE,  #//< planar GBR 4:4:4 48bpp, big-endian
        AV_PIX_FMT_GBRP16LE,  #//< planar GBR 4:4:4 48bpp, little-endian
        AV_PIX_FMT_YUVA422P,  #//< planar YUV 4:2:2 24bpp, (1 Cr & Cb sample per 2x1 Y & A samples)
        AV_PIX_FMT_YUVA444P,  #//< planar YUV 4:4:4 32bpp, (1 Cr & Cb sample per 1x1 Y & A samples)
        AV_PIX_FMT_YUVA420P9BE,  #//< planar YUV 4:2:0 22.5bpp, (1 Cr & Cb sample per 2x2 Y & A samples), big-endian
        AV_PIX_FMT_YUVA420P9LE,  #//< planar YUV 4:2:0 22.5bpp, (1 Cr & Cb sample per 2x2 Y & A samples), little-endian
        AV_PIX_FMT_YUVA422P9BE,  #//< planar YUV 4:2:2 27bpp, (1 Cr & Cb sample per 2x1 Y & A samples), big-endian
        AV_PIX_FMT_YUVA422P9LE,  #//< planar YUV 4:2:2 27bpp, (1 Cr & Cb sample per 2x1 Y & A samples), little-endian
        AV_PIX_FMT_YUVA444P9BE,  #//< planar YUV 4:4:4 36bpp, (1 Cr & Cb sample per 1x1 Y & A samples), big-endian
        AV_PIX_FMT_YUVA444P9LE,  #//< planar YUV 4:4:4 36bpp, (1 Cr & Cb sample per 1x1 Y & A samples), little-endian
        AV_PIX_FMT_YUVA420P10BE, #//< planar YUV 4:2:0 25bpp, (1 Cr & Cb sample per 2x2 Y & A samples, big-endian)
        AV_PIX_FMT_YUVA420P10LE, #//< planar YUV 4:2:0 25bpp, (1 Cr & Cb sample per 2x2 Y & A samples, little-endian)
        AV_PIX_FMT_YUVA422P10BE, #//< planar YUV 4:2:2 30bpp, (1 Cr & Cb sample per 2x1 Y & A samples, big-endian)
        AV_PIX_FMT_YUVA422P10LE, #//< planar YUV 4:2:2 30bpp, (1 Cr & Cb sample per 2x1 Y & A samples, little-endian)
        AV_PIX_FMT_YUVA444P10BE, #//< planar YUV 4:4:4 40bpp, (1 Cr & Cb sample per 1x1 Y & A samples, big-endian)
        AV_PIX_FMT_YUVA444P10LE, #//< planar YUV 4:4:4 40bpp, (1 Cr & Cb sample per 1x1 Y & A samples, little-endian)
        AV_PIX_FMT_YUVA420P16BE, #//< planar YUV 4:2:0 40bpp, (1 Cr & Cb sample per 2x2 Y & A samples, big-endian)
        AV_PIX_FMT_YUVA420P16LE, #//< planar YUV 4:2:0 40bpp, (1 Cr & Cb sample per 2x2 Y & A samples, little-endian)
        AV_PIX_FMT_YUVA422P16BE, #//< planar YUV 4:2:2 48bpp, (1 Cr & Cb sample per 2x1 Y & A samples, big-endian)
        AV_PIX_FMT_YUVA422P16LE, #//< planar YUV 4:2:2 48bpp, (1 Cr & Cb sample per 2x1 Y & A samples, little-endian)
        AV_PIX_FMT_YUVA444P16BE, #//< planar YUV 4:4:4 64bpp, (1 Cr & Cb sample per 1x1 Y & A samples, big-endian)
        AV_PIX_FMT_YUVA444P16LE, #//< planar YUV 4:4:4 64bpp, (1 Cr & Cb sample per 1x1 Y & A samples, little-endian)

        AV_PIX_FMT_VDPAU,         #//< HW acceleration through VDPAU, Picture.data[3] contains a VdpVideoSurface

        AV_PIX_FMT_XYZ12LE,          #//< packed XYZ 4:4:4, 36 bpp, (msb) 12X, 12Y, 12Z (lsb), the 2-byte value for each X/Y/Z is stored as little-endian, the 4 lower bits are set to 0
        AV_PIX_FMT_XYZ12BE,          #//< packed XYZ 4:4:4, 36 bpp, (msb) 12X, 12Y, 12Z (lsb), the 2-byte value for each X/Y/Z is stored as big-endian, the 4 lower bits are set to 0
        AV_PIX_FMT_NV16,                 #//< interleaved chroma YUV 4:2:2, 16bpp, (1 Cr & Cb sample per 2x1 Y samples)
        AV_PIX_FMT_NV20LE,           #//< interleaved chroma YUV 4:2:2, 20bpp, (1 Cr & Cb sample per 2x1 Y samples), little-endian
        AV_PIX_FMT_NV20BE,           #//< interleaved chroma YUV 4:2:2, 20bpp, (1 Cr & Cb sample per 2x1 Y samples), big-endian

        AV_PIX_FMT_RGBA64BE,         #//< packed RGBA 16:16:16:16, 64bpp, 16R, 16G, 16B, 16A, the 2-byte value for each R/G/B/A component is stored as big-endian
        AV_PIX_FMT_RGBA64LE,         #//< packed RGBA 16:16:16:16, 64bpp, 16R, 16G, 16B, 16A, the 2-byte value for each R/G/B/A component is stored as little-endian
        AV_PIX_FMT_BGRA64BE,         #//< packed RGBA 16:16:16:16, 64bpp, 16B, 16G, 16R, 16A, the 2-byte value for each R/G/B/A component is stored as big-endian
        AV_PIX_FMT_BGRA64LE,         #//< packed RGBA 16:16:16:16, 64bpp, 16B, 16G, 16R, 16A, the 2-byte value for each R/G/B/A component is stored as little-endian

        AV_PIX_FMT_YVYU422,   #//< packed YUV 4:2:2, 16bpp, Y0 Cr Y1 Cb

        AV_PIX_FMT_YA16BE,           #//< 16 bits gray, 16 bits alpha (big-endian)
        AV_PIX_FMT_YA16LE,           #//< 16 bits gray, 16 bits alpha (little-endian)

        AV_PIX_FMT_GBRAP,                #//< planar GBRA 4:4:4:4 32bpp
        AV_PIX_FMT_GBRAP16BE,        #//< planar GBRA 4:4:4:4 64bpp, big-endian
        AV_PIX_FMT_GBRAP16LE,        #//< planar GBRA 4:4:4:4 64bpp, little-endian

        AV_PIX_FMT_QSV,

        AV_PIX_FMT_MMAL,

        AV_PIX_FMT_D3D11VA_VLD,  #//< HW decoding through Direct3D11 via old API, Picture.data[3] contains a ID3D11VideoDecoderOutputView pointer


        AV_PIX_FMT_CUDA,

        AV_PIX_FMT_0RGB,                #//< packed RGB 8:8:8, 32bpp, XRGBXRGB...   X=unused/undefined
        AV_PIX_FMT_RGB0,                #//< packed RGB 8:8:8, 32bpp, RGBXRGBX...   X=unused/undefined
        AV_PIX_FMT_0BGR,                #//< packed BGR 8:8:8, 32bpp, XBGRXBGR...   X=unused/undefined
        AV_PIX_FMT_BGR0,                #//< packed BGR 8:8:8, 32bpp, BGRXBGRX...   X=unused/undefined

        AV_PIX_FMT_YUV420P12BE, #//< planar YUV 4:2:0,18bpp, (1 Cr & Cb sample per 2x2 Y samples), big-endian
        AV_PIX_FMT_YUV420P12LE, #//< planar YUV 4:2:0,18bpp, (1 Cr & Cb sample per 2x2 Y samples), little-endian
        AV_PIX_FMT_YUV420P14BE, #//< planar YUV 4:2:0,21bpp, (1 Cr & Cb sample per 2x2 Y samples), big-endian
        AV_PIX_FMT_YUV420P14LE, #//< planar YUV 4:2:0,21bpp, (1 Cr & Cb sample per 2x2 Y samples), little-endian
        AV_PIX_FMT_YUV422P12BE, #//< planar YUV 4:2:2,24bpp, (1 Cr & Cb sample per 2x1 Y samples), big-endian
        AV_PIX_FMT_YUV422P12LE, #//< planar YUV 4:2:2,24bpp, (1 Cr & Cb sample per 2x1 Y samples), little-endian
        AV_PIX_FMT_YUV422P14BE, #//< planar YUV 4:2:2,28bpp, (1 Cr & Cb sample per 2x1 Y samples), big-endian
        AV_PIX_FMT_YUV422P14LE, #//< planar YUV 4:2:2,28bpp, (1 Cr & Cb sample per 2x1 Y samples), little-endian
        AV_PIX_FMT_YUV444P12BE, #//< planar YUV 4:4:4,36bpp, (1 Cr & Cb sample per 1x1 Y samples), big-endian
        AV_PIX_FMT_YUV444P12LE, #//< planar YUV 4:4:4,36bpp, (1 Cr & Cb sample per 1x1 Y samples), little-endian
        AV_PIX_FMT_YUV444P14BE, #//< planar YUV 4:4:4,42bpp, (1 Cr & Cb sample per 1x1 Y samples), big-endian
        AV_PIX_FMT_YUV444P14LE, #//< planar YUV 4:4:4,42bpp, (1 Cr & Cb sample per 1x1 Y samples), little-endian
        AV_PIX_FMT_GBRP12BE,        #//< planar GBR 4:4:4 36bpp, big-endian
        AV_PIX_FMT_GBRP12LE,        #//< planar GBR 4:4:4 36bpp, little-endian
        AV_PIX_FMT_GBRP14BE,        #//< planar GBR 4:4:4 42bpp, big-endian
        AV_PIX_FMT_GBRP14LE,        #//< planar GBR 4:4:4 42bpp, little-endian
        AV_PIX_FMT_YUVJ411P,        #//< planar YUV 4:1:1, 12bpp, (1 Cr & Cb sample per 4x1 Y samples) full scale (JPEG), deprecated in favor of AV_PIX_FMT_YUV411P and setting color_range

        AV_PIX_FMT_BAYER_BGGR8,        #//< bayer, BGBG..(odd line), GRGR..(even line), 8-bit samples */
        AV_PIX_FMT_BAYER_RGGB8,        #//< bayer, RGRG..(odd line), GBGB..(even line), 8-bit samples */
        AV_PIX_FMT_BAYER_GBRG8,        #//< bayer, GBGB..(odd line), RGRG..(even line), 8-bit samples */
        AV_PIX_FMT_BAYER_GRBG8,        #//< bayer, GRGR..(odd line), BGBG..(even line), 8-bit samples */
        AV_PIX_FMT_BAYER_BGGR16LE, #//< bayer, BGBG..(odd line), GRGR..(even line), 16-bit samples, little-endian */
        AV_PIX_FMT_BAYER_BGGR16BE, #//< bayer, BGBG..(odd line), GRGR..(even line), 16-bit samples, big-endian */
        AV_PIX_FMT_BAYER_RGGB16LE, #//< bayer, RGRG..(odd line), GBGB..(even line), 16-bit samples, little-endian */
        AV_PIX_FMT_BAYER_RGGB16BE, #//< bayer, RGRG..(odd line), GBGB..(even line), 16-bit samples, big-endian */
        AV_PIX_FMT_BAYER_GBRG16LE, #//< bayer, GBGB..(odd line), RGRG..(even line), 16-bit samples, little-endian */
        AV_PIX_FMT_BAYER_GBRG16BE, #//< bayer, GBGB..(odd line), RGRG..(even line), 16-bit samples, big-endian */
        AV_PIX_FMT_BAYER_GRBG16LE, #//< bayer, GRGR..(odd line), BGBG..(even line), 16-bit samples, little-endian */
        AV_PIX_FMT_BAYER_GRBG16BE, #//< bayer, GRGR..(odd line), BGBG..(even line), 16-bit samples, big-endian */

        AV_PIX_FMT_XVMC,#//< XVideo Motion Acceleration via common packet passing

        AV_PIX_FMT_YUV440P10LE, #//< planar YUV 4:4:0,20bpp, (1 Cr & Cb sample per 1x2 Y samples), little-endian
        AV_PIX_FMT_YUV440P10BE, #//< planar YUV 4:4:0,20bpp, (1 Cr & Cb sample per 1x2 Y samples), big-endian
        AV_PIX_FMT_YUV440P12LE, #//< planar YUV 4:4:0,24bpp, (1 Cr & Cb sample per 1x2 Y samples), little-endian
        AV_PIX_FMT_YUV440P12BE, #//< planar YUV 4:4:0,24bpp, (1 Cr & Cb sample per 1x2 Y samples), big-endian
        AV_PIX_FMT_AYUV64LE,        #//< packed AYUV 4:4:4,64bpp (1 Cr & Cb sample per 1x1 Y & A samples), little-endian
        AV_PIX_FMT_AYUV64BE,        #//< packed AYUV 4:4:4,64bpp (1 Cr & Cb sample per 1x1 Y & A samples), big-endian

        AV_PIX_FMT_VIDEOTOOLBOX, #//< hardware decoding through Videotoolbox

        AV_PIX_FMT_P010LE, #//< like NV12, with 10bpp per component, data in the high bits, zeros in the low bits, little-endian
        AV_PIX_FMT_P010BE, #//< like NV12, with 10bpp per component, data in the high bits, zeros in the low bits, big-endian

        AV_PIX_FMT_GBRAP12BE,  #//< planar GBR 4:4:4:4 48bpp, big-endian
        AV_PIX_FMT_GBRAP12LE,  #//< planar GBR 4:4:4:4 48bpp, little-endian

        AV_PIX_FMT_GBRAP10BE,  #//< planar GBR 4:4:4:4 40bpp, big-endian
        AV_PIX_FMT_GBRAP10LE,  #//< planar GBR 4:4:4:4 40bpp, little-endian

        AV_PIX_FMT_MEDIACODEC, #//< hardware decoding through MediaCodec

        AV_PIX_FMT_GRAY12BE,   #//<                Y                , 12bpp, big-endian
        AV_PIX_FMT_GRAY12LE,   #//<                Y                , 12bpp, little-endian
        AV_PIX_FMT_GRAY10BE,   #//<                Y                , 10bpp, big-endian
        AV_PIX_FMT_GRAY10LE,   #//<                Y                , 10bpp, little-endian

        AV_PIX_FMT_P016LE, #//< like NV12, with 16bpp per component, little-endian
        AV_PIX_FMT_P016BE, #//< like NV12, with 16bpp per component, big-endian

        AV_PIX_FMT_D3D11,

        AV_PIX_FMT_GRAY9BE,   #//<                Y                , 9bpp, big-endian
        AV_PIX_FMT_GRAY9LE,   #//<                Y                , 9bpp, little-endian

        AV_PIX_FMT_GBRPF32BE,  #//< IEEE-754 single precision planar GBR 4:4:4,         96bpp, big-endian
        AV_PIX_FMT_GBRPF32LE,  #//< IEEE-754 single precision planar GBR 4:4:4,         96bpp, little-endian
        AV_PIX_FMT_GBRAPF32BE, #//< IEEE-754 single precision planar GBRA 4:4:4:4, 128bpp, big-endian
        AV_PIX_FMT_GBRAPF32LE, #//< IEEE-754 single precision planar GBRA 4:4:4:4, 128bpp, little-endian

        AV_PIX_FMT_DRM_PRIME,
        AV_PIX_FMT_OPENCL,

        AV_PIX_FMT_GRAY14BE,   #//<                Y                , 14bpp, big-endian
        AV_PIX_FMT_GRAY14LE,   #//<                Y                , 14bpp, little-endian

        AV_PIX_FMT_GRAYF32BE,  #//< IEEE-754 single precision Y, 32bpp, big-endian
        AV_PIX_FMT_GRAYF32LE,  #//< IEEE-754 single precision Y, 32bpp, little-endian

        AV_PIX_FMT_NB  

#util/dict.h
cdef extern from "../libs/ffmpeg/include/libavutil/dict.h":
    struct AVDictionaryEntry:
        char *key
        char *value

    ctypedef struct AVDictionary
#format/avio.h
cdef extern from "../libs/ffmpeg/include/libavformat/avio.h":
    enum AVIODataMarkerType:
        AVIO_DATA_MARKER_HEADER,
        AVIO_DATA_MARKER_SYNC_POINT,
        AVIO_DATA_MARKER_BOUNDARY_POINT,
        AVIO_DATA_MARKER_UNKNOWN,
        AVIO_DATA_MARKER_TRAILER,
        AVIO_DATA_MARKER_FLUSH_POINT

    struct AVIOInterruptCB:
        int (*callback)(void*)
        void *opaque

    struct AVIOContext:
        const AVClass *av_class
        unsigned char *buffer  
        int buffer_size
        unsigned char *buf_ptr
        unsigned char *buf_end
        void *opaque
        int (*read_packet)(void *opaque, uint8_t *buf, int buf_size)
        int (*write_packet)(void *opaque, uint8_t *buf, int buf_size)
        int64_t (*seek)(void *opaque, int64_t offset, int whence)
        int64_t pos
        int eof_reached
        int write_flag
        int max_packet_size
        unsigned long checksum
        unsigned char *checksum_ptr
        unsigned long (*update_checksum)(unsigned long checksum, const uint8_t *buf, unsigned int size)
        int error
        int (*read_pause)(void *opaque, int pause)
        int64_t (*read_seek)(void *opaque, int stream_index,int64_t timestamp, int flags)
        int seekable
        int64_t maxsize
        int direct
        int64_t bytes_read
        int seek_count
        int writeout_count
        int orig_buffer_size
        int short_seek_threshold
        const char *protocol_whitelist
        const char *protocol_blacklist
        int (*write_data_type)(void *opaque, uint8_t *buf, int buf_size,AVIODataMarkerType type, int64_t time)
        int ignore_boundary_point
        AVIODataMarkerType current_type
        int64_t last_time
        int (*short_seek_get)(void *opaque)
        int64_t written
        unsigned char *buf_ptr_max
        int min_packet_size
#format
cdef extern from "../libs/ffmpeg/include/libavformat/avformat.h":
    ctypedef int (*av_format_control_message)(AVFormatContext *s, int type,void *data, size_t data_size)
    ctypedef struct AVFormatInternal
    void av_register_all();
    struct AVCodecTag:
        pass
    struct AVProbeData:
        const char *filename
        unsigned char *buf
        int buf_size
        const char *mime_type

    enum AVDurationEstimationMethod:
        AVFMT_DURATION_FROM_PTS,    #< Duration accurately estimated from PTSes
        AVFMT_DURATION_FROM_STREAM, #< Duration estimated from a stream with a known duration
        AVFMT_DURATION_FROM_BITRATE #< Duration estimated from bitrate (less accurate)
    enum:
        # for AVFormatContext.flags
        AVFMT_FLAG_GENPTS      = 0x0001     #< Generate missing pts even if it requires parsing future frames.
        AVFMT_FLAG_IGNIDX      = 0x0002     #< Ignore index.
        AVFMT_FLAG_NONBLOCK    = 0x0004     #< Do not block when reading packets from input.
        AVFMT_FLAG_IGNDTS      = 0x0008     #< Ignore DTS on frames that contain both DTS & PTS
        AVFMT_FLAG_NOFILLIN    = 0x0010     #< Do not infer any values from other values, just return what is stored in the container
        AVFMT_FLAG_NOPARSE     = 0x0020     #< Do not use AVParsers, you also must set AVFMT_FLAG_NOFILLIN as the fillin code works on frames and no parsing -> no frames. Also seeking to frames can not work if parsing to find frame boundaries has been disabled
        AVFMT_FLAG_NOBUFFER    = 0x0040     #< Add RTP hinting to the output file
        AVFMT_FLAG_CUSTOM_IO   = 0x0080     #< The caller has supplied a custom AVIOContext, don't avio_close() it.
        AVFMT_FLAG_DISCARD_CORRUPT = 0x0100 #< Discard frames marked corrupted
        AVFMT_FLAG_FLUSH_PACKETS   = 0x0200 #< Flush the AVIOContext every packet.
        AVFMT_FLAG_BITEXACT        = 0x0400
        AVFMT_FLAG_MP4A_LATM   = 0x8000     #< Enable RTP MP4A-LATM payload
        AVFMT_FLAG_SORT_DTS    = 0x10000    #< try to interleave outputted packets by dts (using this flag can slow demuxing down)
        AVFMT_FLAG_PRIV_OPT    = 0x20000    #< Enable use of private options by delaying codec open (this could be made default once all code is converted)
        AVFMT_FLAG_KEEP_SIDE_DATA = 0x40000 #< Don't merge side data but keep it separate.
        AVFMT_FLAG_FAST_SEEK  = 0x80000     #///< Enable fast, but inaccurate seeks for some formats
        AVFMT_FLAG_SHORTEST  = 0x100000     #///< Stop muxing when the shortest stream stops.
        AVFMT_FLAG_AUTO_BSF  = 0x200000     #///< Add bitstream filters as requested by the muxer
        FF_FDEBUG_TS       = 0x0001
        AVFMT_EVENT_FLAG_METADATA_UPDATED  = 0x0001
        AVFMT_AVOID_NEG_TS_AUTO  = -1       #///< Enabled when required by target format
        AVFMT_AVOID_NEG_TS_MAKE_NON_NEGATIVE = 1 #///< Shift timestamps so they are non negative
        AVFMT_AVOID_NEG_TS_MAKE_ZERO   = 2  #///< Shift timestamps so that they start at 0

    struct AVStream:
        int index                        #/**< stream index in AVFormatContext */
        int id                
        AVCodecContext *codec                
        void *priv_data                
        AVRational time_base                
        int64_t start_time                
        int64_t duration                
        int64_t nb_frames                                                 #///< number of frames in this stream if known or 0
        int disposition                 #/**< AV_DISPOSITION_* bit field */
        AVDiscard discard                 #///< Selects which packets can be discarded at will and do not need to be demuxed.
        AVRational sample_aspect_ratio                
        AVDictionary *metadata                
        AVRational avg_frame_rate                
        AVPacket attached_pic                
        AVPacketSideData *side_data                
        int event_flags                
        AVRational r_frame_rate                
        char *recommended_encoder_configuration                
        AVCodecParameters *codecpar                

    struct AVChapter:
        int id                              #///< unique ID to identify the chapter
        AVRational time_base                #///< time base in which the start/end timestamps are specified
        int64_t start, end                  #///< chapter start/end time in time_base units
        AVDictionary *metadata

    struct AVProgram:
        int            id        
        int            flags        
        AVDiscard discard               
        unsigned int   *stream_index        
        unsigned int   nb_stream_indexes        
        AVDictionary *metadata        

        int program_num        
        int pmt_pid        
        int pcr_pid        
        int pmt_version        
        int64_t start_time        
        int64_t end_time        

        int64_t pts_wrap_reference           
        int pts_wrap_behavior                

    struct AVInputFormat:
        const char *name            #< A comma separated list of short names for the format
        const char *long_name       #< Descriptive name for the format, meant to be more human-readable than name  
        # * Can use flags: AVFMT_NOFILE, AVFMT_NEEDNUMBER, AVFMT_SHOW_IDS,
        # * AVFMT_GENERIC_INDEX, AVFMT_TS_DISCONT, AVFMT_NOBINSEARCH,
        # * AVFMT_NOGENSEARCH, AVFMT_NO_BYTE_SEEK, AVFMT_SEEK_TO_PTS.
        int flags
        const char *extensions
        AVCodecTag **codec_tag
        const AVClass *priv_class
        const char *mime_type       #< Comma-separated list of mime types
        AVInputFormat *next
        int raw_codec_id
        int priv_data_size
        int (*read_probe)(AVProbeData *)
        int (*read_header)(AVFormatContext *)
        int (*read_packet)(AVFormatContext *, AVPacket *pkt)
        int (*read_close)(AVFormatContext *)
        int (*read_seek)(AVFormatContext *,int stream_index, int64_t timestamp, int flags)
        int64_t (*read_timestamp)(AVFormatContext *s, int stream_index,int64_t *pos, int64_t pos_limit)
        int (*read_play)(AVFormatContext *)
        int (*read_pause)(AVFormatContext *)
        int (*read_seek2)(AVFormatContext *s, int stream_index, int64_t min_ts, int64_t ts, int64_t max_ts, int flags)
        int (*get_device_list)(AVFormatContext *s, AVDeviceInfoList *device_list)
        int (*create_device_capabilities)(AVFormatContext *s, AVDeviceCapabilitiesQuery *caps)
        int (*free_device_capabilities)(AVFormatContext *s, AVDeviceCapabilitiesQuery *caps)
    

    struct AVOutputFormat:
        const char *name
        const char *long_name
        const char *mime_type
        const char *extensions
        AVCodecID audio_codec       #< default audio codec
        AVCodecID video_codec       #< default video codec
        AVCodecID subtitle_codec    #< default subtitle codec
        # * can use flags: AVFMT_NOFILE, AVFMT_NEEDNUMBER, AVFMT_RAWPICTURE,
        # * AVFMT_GLOBALHEADER, AVFMT_NOTIMESTAMPS, AVFMT_VARIABLE_FPS,
        # * AVFMT_NODIMENSIONS, AVFMT_NOSTREAMS, AVFMT_ALLOW_FLUSH,
        # * AVFMT_TS_NONSTRICT
        int flags
        AVCodecTag **codec_tag
        const AVClass *priv_class
        AVOutputFormat *next
        int priv_data_size
        int (*write_header)(AVFormatContext *)
        int (*write_packet)(AVFormatContext *, AVPacket *pkt)
        int (*write_trailer)(AVFormatContext *)
        int (*interleave_packet)(AVFormatContext *, AVPacket *, AVPacket *, int flush)
        int (*query_codec)(AVCodecID id, int std_compliance)
        void (*get_output_timestamp)(AVFormatContext *s, int stream,int64_t *dts, int64_t *wall)
        int (*control_message)(AVFormatContext *s, int type,void *data, size_t data_size)
        int (*write_uncoded_frame)(AVFormatContext *, int stream_index, AVFrame **frame, unsigned flags)
        int (*get_device_list)(AVFormatContext *s, AVDeviceInfoList *device_list)
        int (*create_device_capabilities)(AVFormatContext *s, AVDeviceCapabilitiesQuery *caps)
        int (*free_device_capabilities)(AVFormatContext *s, AVDeviceCapabilitiesQuery *caps)
        AVCodecID data_codec
        int (*init)( AVFormatContext *)
        void (*deinit)(AVFormatContext *)
        int (*check_bitstream)(AVFormatContext *, const AVPacket *pkt)

    struct AVFormatContext:
        const AVClass       av_class
        AVInputFormat *     iformat         # The input container format
        AVOutputFormat *    oformat         # The output container format
        void *              priv_data       # Format private data
        AVIOContext *       pb              # I/O context
        int                 ctx_flags       # stream info, see AVFMTCTX_
        unsigned int        nb_streams      # Number of elements in AVFormatContext.streams
        AVStream            **streams       # A list of all streams in the file
#        char                filename[1024]  # input or output filename
        char                *url
        int64_t             start_time      # Position of the first frame of the component, in AV_TIME_BASE fractional seconds
        int64_t             duration        # Duration of the stream, in AV_TIME_BASE fractional seconds
        int64_t                 bit_rate        # Total stream bitrate in bit/s, 0 if not available
        unsigned int        packet_size
        int                 max_delay
        int                 flags           # Flags modifying the (de)muxer behaviour. A combination of AVFMT_FLAG_*
        int64_t             probesize       # deprecated in favor of probesize2
        int64_t             max_analyze_duration # deprecated in favor of max_analyze_duration2
        const uint8_t       *key
        int                 keylen
        unsigned int        nb_programs
        AVProgram           **programs
        AVCodecID           video_codec_id  # Forced video codec_id
        AVCodecID           audio_codec_id  # Forced audio codec_id
        AVCodecID           subtitle_codec_id # Forced subtitle codec_id
        unsigned int        max_index_size  # Maximum amount of memory in bytes to use for the index of each stream
        unsigned int        max_picture_buffer # Maximum amount of memory in bytes to use for buffering frames
        unsigned int        nb_chapters     # Number of chapters in AVChapter array
        AVChapter           **chapters
        AVDictionary        *metadata       # Metadata that applies to the whole file
        int64_t             start_time_realtime # Start time of the stream in real world time, in microseconds since the Unix epoch
        int                 fps_probe_size  # The number of frames used for determining the framerate in avformat_find_stream_info()
        int                 error_recognition # Error recognition; higher values will detect more errors
        AVIOInterruptCB     interrupt_callback # Custom interrupt callbacks for the I/O layer
        int                 debug           # Flags to enable debugging
        int64_t             max_interleave_delta # Maximum buffering duration for interleaving
        int                 strict_std_compliance # Allow non-standard and experimental extension
        int                 event_flags     # Flags for the user to detect events happening on the file
        int                 max_ts_probe    # Maximum number of packets to read while waiting for the first timestamp
        int                 avoid_negative_ts # Avoid negative timestamps during muxing, see AVFMT_AVOID_NEG_TS_*
        int                 ts_id           # Transport stream id
        int                 audio_preload   # Audio preload in microseconds
        int                 max_chunk_duration # Max chunk time in microseconds
        int                 max_chunk_size  # Max chunk size in bytes
        int                 use_wallclock_as_timestamps # forces the use of wallclock timestamps as pts/dts of packets
        int                 avio_flags      # avio flags
        AVDurationEstimationMethod duration_estimation_method
        int64_t             skip_initial_bytes # Skip initial bytes when opening stream
        unsigned int        correct_ts_overflow # Correct single timestamp overflows
        int                 seek2any        # Force seeking to any (also non key) frames
        int                 flush_packets   # Flush the I/O context after each packet
        int                 probe_score     # format probing score
        int                 format_probesize # number of bytes to read maximally to identify format
        char                *codec_whitelist # ',' separated list of allowed decoders
        char                *format_whitelist # ',' separated list of allowed demuxers
        AVFormatInternal    *internal       # An opaque field for libavformat internal usage
        int                 io_repositioned # IO repositioned flag
        AVCodec             *video_codec    # Forced video codec
        AVCodec             *audio_codec    # Forced audio codec
        AVCodec             *subtitle_codec # Forced subtitle codec
        AVCodec             *data_codec     # Forced data codec
        int                 metadata_header_padding # Number of bytes to be written as padding in a metadata header
        void                *opaque         # User data
        # Callback used by devices to communicate with application
        #int (*control_message_cb)(AVFormatContext *s, int type, void *data, size_t data_size)
        av_format_control_message control_message_cb
        int64_t             output_ts_offset # Output timestamp offset, in microseconds
        #int64_t             max_analyze_duration2 # Maximum duration (in AV_TIME_BASE units) of the data read from input in avformat_find_stream_info()
        #int64_t             probesize2      # Maximum size of the data read from input for determining the input container format
        uint8_t             *dump_separator # dump format separator
        AVCodecID           data_codec_id   # Forced Data codec_id
        char                *protocol_whitelist
        int (*io_open)( AVFormatContext *s, AVIOContext **pb, const char *url,int flags, AVDictionary **options)
        void (*io_close)( AVFormatContext *s, AVIOContext *pb)
        char                *protocol_blacklist
        int                 max_streams
        int                 skip_estimate_duration_from_pt
#cdef extern from "string.h":
#    memcpy(void * dst, void * src, unsigned long sz)
#    memset(void * dst, unsigned char c, unsigned long sz)
#device
cdef extern from "../libs/ffmpeg/include/libavdevice/avdevice.h":
    #设备信息
    struct AVDeviceInfo:
        char *device_name
        char *device_description
    struct AVDeviceInfoList:
        AVDeviceInfo **devices
        int nb_devices
        int default_device

    struct AVDeviceCapabilitiesQuery:
        const AVClass *av_class
        AVFormatContext *device_context
        AVCodecID codec
        AVSampleFormat sample_format
        AVPixelFormat pixel_format
        int sample_rate
        int channels
        int64_t channel_layout
        int window_width
        int window_height
        int frame_width
        int frame_height
        AVRational fps

    #设备查找的方法
    #列出所有可用的设备及其信息
    int avdevice_list_devices(AVFormatContext *s, AVDeviceInfoList **device_list)
    #清除由avdevice_list_devices列出的设备缓存
    void avdevice_free_list_devices(AVDeviceInfoList **device_list)
    #列出输入设备
    int avdevice_list_input_sources(AVInputFormat *device, const char *device_name,
                                AVDictionary *device_options, AVDeviceInfoList **device_list)
    #列出输出设备
    int avdevice_list_output_sinks(AVOutputFormat *device, const char *device_name,
                               AVDictionary *device_options, AVDeviceInfoList **device_list)
    #注册所有的设备，初始化时使用。    
    void avdevice_register_all()
    #输出下一个可用的音频输入设备
    AVInputFormat *av_input_audio_device_next(AVInputFormat  *d)
    #输出下一个可用的视频输入设备
    AVInputFormat *av_input_video_device_next(AVInputFormat  *d)
    #输出下一个可用的音频输出设备
    AVOutputFormat *av_output_audio_device_next(AVOutputFormat *d)
    #输出下一个可用的视频输出设备
    AVOutputFormat *av_output_video_device_next(AVOutputFormat *d)

#######################################
#以下为自定义的处理类
#######################################
cdef AVRational AV_TIME_BASE_Q
AV_TIME_BASE_Q.num = 1
AV_TIME_BASE_Q.den = AV_TIME_BASE
