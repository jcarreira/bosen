// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: ccdmf.proto

#ifndef PROTOBUF_ccdmf_2eproto__INCLUDED
#define PROTOBUF_ccdmf_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 2006000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 2006000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/generated_enum_reflection.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace stradsccdmf {

// Internal implementation detail -- do not call these.
void  protobuf_AddDesc_ccdmf_2eproto();
void protobuf_AssignDesc_ccdmf_2eproto();
void protobuf_ShutdownFile_ccdmf_2eproto();

class initinfo;
class singleint;
class singlebucket;
class topicrow;
class controlmsg;

enum controlmsg_msgtype {
  controlmsg_msgtype_BUCKET = 1,
  controlmsg_msgtype_EXITRING = 2,
  controlmsg_msgtype_UPDATE = 3,
  controlmsg_msgtype_OBJSYNC = 4,
  controlmsg_msgtype_UNKNOWN = 5
};
bool controlmsg_msgtype_IsValid(int value);
const controlmsg_msgtype controlmsg_msgtype_msgtype_MIN = controlmsg_msgtype_BUCKET;
const controlmsg_msgtype controlmsg_msgtype_msgtype_MAX = controlmsg_msgtype_UNKNOWN;
const int controlmsg_msgtype_msgtype_ARRAYSIZE = controlmsg_msgtype_msgtype_MAX + 1;

const ::google::protobuf::EnumDescriptor* controlmsg_msgtype_descriptor();
inline const ::std::string& controlmsg_msgtype_Name(controlmsg_msgtype value) {
  return ::google::protobuf::internal::NameOfEnum(
    controlmsg_msgtype_descriptor(), value);
}
inline bool controlmsg_msgtype_Parse(
    const ::std::string& name, controlmsg_msgtype* value) {
  return ::google::protobuf::internal::ParseNamedEnum<controlmsg_msgtype>(
    controlmsg_msgtype_descriptor(), name, value);
}
// ===================================================================

class initinfo : public ::google::protobuf::Message {
 public:
  initinfo();
  virtual ~initinfo();

  initinfo(const initinfo& from);

  inline initinfo& operator=(const initinfo& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const initinfo& default_instance();

  void Swap(initinfo* other);

  // implements Message ----------------------------------------------

  initinfo* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const initinfo& from);
  void MergeFrom(const initinfo& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required int32 tokencnt = 1;
  inline bool has_tokencnt() const;
  inline void clear_tokencnt();
  static const int kTokencntFieldNumber = 1;
  inline ::google::protobuf::int32 tokencnt() const;
  inline void set_tokencnt(::google::protobuf::int32 value);

  // required int32 docnt = 2;
  inline bool has_docnt() const;
  inline void clear_docnt();
  static const int kDocntFieldNumber = 2;
  inline ::google::protobuf::int32 docnt() const;
  inline void set_docnt(::google::protobuf::int32 value);

  // required int32 wordmax = 3;
  inline bool has_wordmax() const;
  inline void clear_wordmax();
  static const int kWordmaxFieldNumber = 3;
  inline ::google::protobuf::int32 wordmax() const;
  inline void set_wordmax(::google::protobuf::int32 value);

  // @@protoc_insertion_point(class_scope:stradsccdmf.initinfo)
 private:
  inline void set_has_tokencnt();
  inline void clear_has_tokencnt();
  inline void set_has_docnt();
  inline void clear_has_docnt();
  inline void set_has_wordmax();
  inline void clear_has_wordmax();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::google::protobuf::int32 tokencnt_;
  ::google::protobuf::int32 docnt_;
  ::google::protobuf::int32 wordmax_;
  friend void  protobuf_AddDesc_ccdmf_2eproto();
  friend void protobuf_AssignDesc_ccdmf_2eproto();
  friend void protobuf_ShutdownFile_ccdmf_2eproto();

  void InitAsDefaultInstance();
  static initinfo* default_instance_;
};
// -------------------------------------------------------------------

class singleint : public ::google::protobuf::Message {
 public:
  singleint();
  virtual ~singleint();

  singleint(const singleint& from);

  inline singleint& operator=(const singleint& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const singleint& default_instance();

  void Swap(singleint* other);

  // implements Message ----------------------------------------------

  singleint* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const singleint& from);
  void MergeFrom(const singleint& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required int32 ivalue = 1;
  inline bool has_ivalue() const;
  inline void clear_ivalue();
  static const int kIvalueFieldNumber = 1;
  inline ::google::protobuf::int32 ivalue() const;
  inline void set_ivalue(::google::protobuf::int32 value);

  // @@protoc_insertion_point(class_scope:stradsccdmf.singleint)
 private:
  inline void set_has_ivalue();
  inline void clear_has_ivalue();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::google::protobuf::int32 ivalue_;
  friend void  protobuf_AddDesc_ccdmf_2eproto();
  friend void protobuf_AssignDesc_ccdmf_2eproto();
  friend void protobuf_ShutdownFile_ccdmf_2eproto();

  void InitAsDefaultInstance();
  static singleint* default_instance_;
};
// -------------------------------------------------------------------

class singlebucket : public ::google::protobuf::Message {
 public:
  singlebucket();
  virtual ~singlebucket();

  singlebucket(const singlebucket& from);

  inline singlebucket& operator=(const singlebucket& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const singlebucket& default_instance();

  void Swap(singlebucket* other);

  // implements Message ----------------------------------------------

  singlebucket* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const singlebucket& from);
  void MergeFrom(const singlebucket& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated int32 wid = 1 [packed = true];
  inline int wid_size() const;
  inline void clear_wid();
  static const int kWidFieldNumber = 1;
  inline ::google::protobuf::int32 wid(int index) const;
  inline void set_wid(int index, ::google::protobuf::int32 value);
  inline void add_wid(::google::protobuf::int32 value);
  inline const ::google::protobuf::RepeatedField< ::google::protobuf::int32 >&
      wid() const;
  inline ::google::protobuf::RepeatedField< ::google::protobuf::int32 >*
      mutable_wid();

  // @@protoc_insertion_point(class_scope:stradsccdmf.singlebucket)
 private:

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::google::protobuf::RepeatedField< ::google::protobuf::int32 > wid_;
  mutable int _wid_cached_byte_size_;
  friend void  protobuf_AddDesc_ccdmf_2eproto();
  friend void protobuf_AssignDesc_ccdmf_2eproto();
  friend void protobuf_ShutdownFile_ccdmf_2eproto();

  void InitAsDefaultInstance();
  static singlebucket* default_instance_;
};
// -------------------------------------------------------------------

class topicrow : public ::google::protobuf::Message {
 public:
  topicrow();
  virtual ~topicrow();

  topicrow(const topicrow& from);

  inline topicrow& operator=(const topicrow& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const topicrow& default_instance();

  void Swap(topicrow* other);

  // implements Message ----------------------------------------------

  topicrow* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const topicrow& from);
  void MergeFrom(const topicrow& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional int32 wordid = 1;
  inline bool has_wordid() const;
  inline void clear_wordid();
  static const int kWordidFieldNumber = 1;
  inline ::google::protobuf::int32 wordid() const;
  inline void set_wordid(::google::protobuf::int32 value);

  // repeated int32 tid = 2 [packed = true];
  inline int tid_size() const;
  inline void clear_tid();
  static const int kTidFieldNumber = 2;
  inline ::google::protobuf::int32 tid(int index) const;
  inline void set_tid(int index, ::google::protobuf::int32 value);
  inline void add_tid(::google::protobuf::int32 value);
  inline const ::google::protobuf::RepeatedField< ::google::protobuf::int32 >&
      tid() const;
  inline ::google::protobuf::RepeatedField< ::google::protobuf::int32 >*
      mutable_tid();

  // repeated int32 assigned = 3 [packed = true];
  inline int assigned_size() const;
  inline void clear_assigned();
  static const int kAssignedFieldNumber = 3;
  inline ::google::protobuf::int32 assigned(int index) const;
  inline void set_assigned(int index, ::google::protobuf::int32 value);
  inline void add_assigned(::google::protobuf::int32 value);
  inline const ::google::protobuf::RepeatedField< ::google::protobuf::int32 >&
      assigned() const;
  inline ::google::protobuf::RepeatedField< ::google::protobuf::int32 >*
      mutable_assigned();

  // @@protoc_insertion_point(class_scope:stradsccdmf.topicrow)
 private:
  inline void set_has_wordid();
  inline void clear_has_wordid();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::google::protobuf::RepeatedField< ::google::protobuf::int32 > tid_;
  mutable int _tid_cached_byte_size_;
  ::google::protobuf::RepeatedField< ::google::protobuf::int32 > assigned_;
  mutable int _assigned_cached_byte_size_;
  ::google::protobuf::int32 wordid_;
  friend void  protobuf_AddDesc_ccdmf_2eproto();
  friend void protobuf_AssignDesc_ccdmf_2eproto();
  friend void protobuf_ShutdownFile_ccdmf_2eproto();

  void InitAsDefaultInstance();
  static topicrow* default_instance_;
};
// -------------------------------------------------------------------

class controlmsg : public ::google::protobuf::Message {
 public:
  controlmsg();
  virtual ~controlmsg();

  controlmsg(const controlmsg& from);

  inline controlmsg& operator=(const controlmsg& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const controlmsg& default_instance();

  void Swap(controlmsg* other);

  // implements Message ----------------------------------------------

  controlmsg* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const controlmsg& from);
  void MergeFrom(const controlmsg& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  typedef controlmsg_msgtype msgtype;
  static const msgtype BUCKET = controlmsg_msgtype_BUCKET;
  static const msgtype EXITRING = controlmsg_msgtype_EXITRING;
  static const msgtype UPDATE = controlmsg_msgtype_UPDATE;
  static const msgtype OBJSYNC = controlmsg_msgtype_OBJSYNC;
  static const msgtype UNKNOWN = controlmsg_msgtype_UNKNOWN;
  static inline bool msgtype_IsValid(int value) {
    return controlmsg_msgtype_IsValid(value);
  }
  static const msgtype msgtype_MIN =
    controlmsg_msgtype_msgtype_MIN;
  static const msgtype msgtype_MAX =
    controlmsg_msgtype_msgtype_MAX;
  static const int msgtype_ARRAYSIZE =
    controlmsg_msgtype_msgtype_ARRAYSIZE;
  static inline const ::google::protobuf::EnumDescriptor*
  msgtype_descriptor() {
    return controlmsg_msgtype_descriptor();
  }
  static inline const ::std::string& msgtype_Name(msgtype value) {
    return controlmsg_msgtype_Name(value);
  }
  static inline bool msgtype_Parse(const ::std::string& name,
      msgtype* value) {
    return controlmsg_msgtype_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  // optional .stradsccdmf.controlmsg.msgtype type = 1 [default = UNKNOWN];
  inline bool has_type() const;
  inline void clear_type();
  static const int kTypeFieldNumber = 1;
  inline ::stradsccdmf::controlmsg_msgtype type() const;
  inline void set_type(::stradsccdmf::controlmsg_msgtype value);

  // optional int32 ringsrc = 2;
  inline bool has_ringsrc() const;
  inline void clear_ringsrc();
  static const int kRingsrcFieldNumber = 2;
  inline ::google::protobuf::int32 ringsrc() const;
  inline void set_ringsrc(::google::protobuf::int32 value);

  // optional int32 ringdst = 3;
  inline bool has_ringdst() const;
  inline void clear_ringdst();
  static const int kRingdstFieldNumber = 3;
  inline ::google::protobuf::int32 ringdst() const;
  inline void set_ringdst(::google::protobuf::int32 value);

  // optional int32 rows = 4;
  inline bool has_rows() const;
  inline void clear_rows();
  static const int kRowsFieldNumber = 4;
  inline ::google::protobuf::int32 rows() const;
  inline void set_rows(::google::protobuf::int32 value);

  // optional int32 cols = 5;
  inline bool has_cols() const;
  inline void clear_cols();
  static const int kColsFieldNumber = 5;
  inline ::google::protobuf::int32 cols() const;
  inline void set_cols(::google::protobuf::int32 value);

  // repeated .stradsccdmf.initinfo init = 6;
  inline int init_size() const;
  inline void clear_init();
  static const int kInitFieldNumber = 6;
  inline const ::stradsccdmf::initinfo& init(int index) const;
  inline ::stradsccdmf::initinfo* mutable_init(int index);
  inline ::stradsccdmf::initinfo* add_init();
  inline const ::google::protobuf::RepeatedPtrField< ::stradsccdmf::initinfo >&
      init() const;
  inline ::google::protobuf::RepeatedPtrField< ::stradsccdmf::initinfo >*
      mutable_init();

  // repeated .stradsccdmf.singlebucket buckets = 7;
  inline int buckets_size() const;
  inline void clear_buckets();
  static const int kBucketsFieldNumber = 7;
  inline const ::stradsccdmf::singlebucket& buckets(int index) const;
  inline ::stradsccdmf::singlebucket* mutable_buckets(int index);
  inline ::stradsccdmf::singlebucket* add_buckets();
  inline const ::google::protobuf::RepeatedPtrField< ::stradsccdmf::singlebucket >&
      buckets() const;
  inline ::google::protobuf::RepeatedPtrField< ::stradsccdmf::singlebucket >*
      mutable_buckets();

  // optional double partialobj = 8;
  inline bool has_partialobj() const;
  inline void clear_partialobj();
  static const int kPartialobjFieldNumber = 8;
  inline double partialobj() const;
  inline void set_partialobj(double value);

  // @@protoc_insertion_point(class_scope:stradsccdmf.controlmsg)
 private:
  inline void set_has_type();
  inline void clear_has_type();
  inline void set_has_ringsrc();
  inline void clear_has_ringsrc();
  inline void set_has_ringdst();
  inline void clear_has_ringdst();
  inline void set_has_rows();
  inline void clear_has_rows();
  inline void set_has_cols();
  inline void clear_has_cols();
  inline void set_has_partialobj();
  inline void clear_has_partialobj();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  int type_;
  ::google::protobuf::int32 ringsrc_;
  ::google::protobuf::int32 ringdst_;
  ::google::protobuf::int32 rows_;
  ::google::protobuf::RepeatedPtrField< ::stradsccdmf::initinfo > init_;
  ::google::protobuf::RepeatedPtrField< ::stradsccdmf::singlebucket > buckets_;
  double partialobj_;
  ::google::protobuf::int32 cols_;
  friend void  protobuf_AddDesc_ccdmf_2eproto();
  friend void protobuf_AssignDesc_ccdmf_2eproto();
  friend void protobuf_ShutdownFile_ccdmf_2eproto();

  void InitAsDefaultInstance();
  static controlmsg* default_instance_;
};
// ===================================================================


// ===================================================================

// initinfo

// required int32 tokencnt = 1;
inline bool initinfo::has_tokencnt() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void initinfo::set_has_tokencnt() {
  _has_bits_[0] |= 0x00000001u;
}
inline void initinfo::clear_has_tokencnt() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void initinfo::clear_tokencnt() {
  tokencnt_ = 0;
  clear_has_tokencnt();
}
inline ::google::protobuf::int32 initinfo::tokencnt() const {
  // @@protoc_insertion_point(field_get:stradsccdmf.initinfo.tokencnt)
  return tokencnt_;
}
inline void initinfo::set_tokencnt(::google::protobuf::int32 value) {
  set_has_tokencnt();
  tokencnt_ = value;
  // @@protoc_insertion_point(field_set:stradsccdmf.initinfo.tokencnt)
}

// required int32 docnt = 2;
inline bool initinfo::has_docnt() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void initinfo::set_has_docnt() {
  _has_bits_[0] |= 0x00000002u;
}
inline void initinfo::clear_has_docnt() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void initinfo::clear_docnt() {
  docnt_ = 0;
  clear_has_docnt();
}
inline ::google::protobuf::int32 initinfo::docnt() const {
  // @@protoc_insertion_point(field_get:stradsccdmf.initinfo.docnt)
  return docnt_;
}
inline void initinfo::set_docnt(::google::protobuf::int32 value) {
  set_has_docnt();
  docnt_ = value;
  // @@protoc_insertion_point(field_set:stradsccdmf.initinfo.docnt)
}

// required int32 wordmax = 3;
inline bool initinfo::has_wordmax() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void initinfo::set_has_wordmax() {
  _has_bits_[0] |= 0x00000004u;
}
inline void initinfo::clear_has_wordmax() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void initinfo::clear_wordmax() {
  wordmax_ = 0;
  clear_has_wordmax();
}
inline ::google::protobuf::int32 initinfo::wordmax() const {
  // @@protoc_insertion_point(field_get:stradsccdmf.initinfo.wordmax)
  return wordmax_;
}
inline void initinfo::set_wordmax(::google::protobuf::int32 value) {
  set_has_wordmax();
  wordmax_ = value;
  // @@protoc_insertion_point(field_set:stradsccdmf.initinfo.wordmax)
}

// -------------------------------------------------------------------

// singleint

// required int32 ivalue = 1;
inline bool singleint::has_ivalue() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void singleint::set_has_ivalue() {
  _has_bits_[0] |= 0x00000001u;
}
inline void singleint::clear_has_ivalue() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void singleint::clear_ivalue() {
  ivalue_ = 0;
  clear_has_ivalue();
}
inline ::google::protobuf::int32 singleint::ivalue() const {
  // @@protoc_insertion_point(field_get:stradsccdmf.singleint.ivalue)
  return ivalue_;
}
inline void singleint::set_ivalue(::google::protobuf::int32 value) {
  set_has_ivalue();
  ivalue_ = value;
  // @@protoc_insertion_point(field_set:stradsccdmf.singleint.ivalue)
}

// -------------------------------------------------------------------

// singlebucket

// repeated int32 wid = 1 [packed = true];
inline int singlebucket::wid_size() const {
  return wid_.size();
}
inline void singlebucket::clear_wid() {
  wid_.Clear();
}
inline ::google::protobuf::int32 singlebucket::wid(int index) const {
  // @@protoc_insertion_point(field_get:stradsccdmf.singlebucket.wid)
  return wid_.Get(index);
}
inline void singlebucket::set_wid(int index, ::google::protobuf::int32 value) {
  wid_.Set(index, value);
  // @@protoc_insertion_point(field_set:stradsccdmf.singlebucket.wid)
}
inline void singlebucket::add_wid(::google::protobuf::int32 value) {
  wid_.Add(value);
  // @@protoc_insertion_point(field_add:stradsccdmf.singlebucket.wid)
}
inline const ::google::protobuf::RepeatedField< ::google::protobuf::int32 >&
singlebucket::wid() const {
  // @@protoc_insertion_point(field_list:stradsccdmf.singlebucket.wid)
  return wid_;
}
inline ::google::protobuf::RepeatedField< ::google::protobuf::int32 >*
singlebucket::mutable_wid() {
  // @@protoc_insertion_point(field_mutable_list:stradsccdmf.singlebucket.wid)
  return &wid_;
}

// -------------------------------------------------------------------

// topicrow

// optional int32 wordid = 1;
inline bool topicrow::has_wordid() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void topicrow::set_has_wordid() {
  _has_bits_[0] |= 0x00000001u;
}
inline void topicrow::clear_has_wordid() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void topicrow::clear_wordid() {
  wordid_ = 0;
  clear_has_wordid();
}
inline ::google::protobuf::int32 topicrow::wordid() const {
  // @@protoc_insertion_point(field_get:stradsccdmf.topicrow.wordid)
  return wordid_;
}
inline void topicrow::set_wordid(::google::protobuf::int32 value) {
  set_has_wordid();
  wordid_ = value;
  // @@protoc_insertion_point(field_set:stradsccdmf.topicrow.wordid)
}

// repeated int32 tid = 2 [packed = true];
inline int topicrow::tid_size() const {
  return tid_.size();
}
inline void topicrow::clear_tid() {
  tid_.Clear();
}
inline ::google::protobuf::int32 topicrow::tid(int index) const {
  // @@protoc_insertion_point(field_get:stradsccdmf.topicrow.tid)
  return tid_.Get(index);
}
inline void topicrow::set_tid(int index, ::google::protobuf::int32 value) {
  tid_.Set(index, value);
  // @@protoc_insertion_point(field_set:stradsccdmf.topicrow.tid)
}
inline void topicrow::add_tid(::google::protobuf::int32 value) {
  tid_.Add(value);
  // @@protoc_insertion_point(field_add:stradsccdmf.topicrow.tid)
}
inline const ::google::protobuf::RepeatedField< ::google::protobuf::int32 >&
topicrow::tid() const {
  // @@protoc_insertion_point(field_list:stradsccdmf.topicrow.tid)
  return tid_;
}
inline ::google::protobuf::RepeatedField< ::google::protobuf::int32 >*
topicrow::mutable_tid() {
  // @@protoc_insertion_point(field_mutable_list:stradsccdmf.topicrow.tid)
  return &tid_;
}

// repeated int32 assigned = 3 [packed = true];
inline int topicrow::assigned_size() const {
  return assigned_.size();
}
inline void topicrow::clear_assigned() {
  assigned_.Clear();
}
inline ::google::protobuf::int32 topicrow::assigned(int index) const {
  // @@protoc_insertion_point(field_get:stradsccdmf.topicrow.assigned)
  return assigned_.Get(index);
}
inline void topicrow::set_assigned(int index, ::google::protobuf::int32 value) {
  assigned_.Set(index, value);
  // @@protoc_insertion_point(field_set:stradsccdmf.topicrow.assigned)
}
inline void topicrow::add_assigned(::google::protobuf::int32 value) {
  assigned_.Add(value);
  // @@protoc_insertion_point(field_add:stradsccdmf.topicrow.assigned)
}
inline const ::google::protobuf::RepeatedField< ::google::protobuf::int32 >&
topicrow::assigned() const {
  // @@protoc_insertion_point(field_list:stradsccdmf.topicrow.assigned)
  return assigned_;
}
inline ::google::protobuf::RepeatedField< ::google::protobuf::int32 >*
topicrow::mutable_assigned() {
  // @@protoc_insertion_point(field_mutable_list:stradsccdmf.topicrow.assigned)
  return &assigned_;
}

// -------------------------------------------------------------------

// controlmsg

// optional .stradsccdmf.controlmsg.msgtype type = 1 [default = UNKNOWN];
inline bool controlmsg::has_type() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void controlmsg::set_has_type() {
  _has_bits_[0] |= 0x00000001u;
}
inline void controlmsg::clear_has_type() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void controlmsg::clear_type() {
  type_ = 5;
  clear_has_type();
}
inline ::stradsccdmf::controlmsg_msgtype controlmsg::type() const {
  // @@protoc_insertion_point(field_get:stradsccdmf.controlmsg.type)
  return static_cast< ::stradsccdmf::controlmsg_msgtype >(type_);
}
inline void controlmsg::set_type(::stradsccdmf::controlmsg_msgtype value) {
  assert(::stradsccdmf::controlmsg_msgtype_IsValid(value));
  set_has_type();
  type_ = value;
  // @@protoc_insertion_point(field_set:stradsccdmf.controlmsg.type)
}

// optional int32 ringsrc = 2;
inline bool controlmsg::has_ringsrc() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void controlmsg::set_has_ringsrc() {
  _has_bits_[0] |= 0x00000002u;
}
inline void controlmsg::clear_has_ringsrc() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void controlmsg::clear_ringsrc() {
  ringsrc_ = 0;
  clear_has_ringsrc();
}
inline ::google::protobuf::int32 controlmsg::ringsrc() const {
  // @@protoc_insertion_point(field_get:stradsccdmf.controlmsg.ringsrc)
  return ringsrc_;
}
inline void controlmsg::set_ringsrc(::google::protobuf::int32 value) {
  set_has_ringsrc();
  ringsrc_ = value;
  // @@protoc_insertion_point(field_set:stradsccdmf.controlmsg.ringsrc)
}

// optional int32 ringdst = 3;
inline bool controlmsg::has_ringdst() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void controlmsg::set_has_ringdst() {
  _has_bits_[0] |= 0x00000004u;
}
inline void controlmsg::clear_has_ringdst() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void controlmsg::clear_ringdst() {
  ringdst_ = 0;
  clear_has_ringdst();
}
inline ::google::protobuf::int32 controlmsg::ringdst() const {
  // @@protoc_insertion_point(field_get:stradsccdmf.controlmsg.ringdst)
  return ringdst_;
}
inline void controlmsg::set_ringdst(::google::protobuf::int32 value) {
  set_has_ringdst();
  ringdst_ = value;
  // @@protoc_insertion_point(field_set:stradsccdmf.controlmsg.ringdst)
}

// optional int32 rows = 4;
inline bool controlmsg::has_rows() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void controlmsg::set_has_rows() {
  _has_bits_[0] |= 0x00000008u;
}
inline void controlmsg::clear_has_rows() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void controlmsg::clear_rows() {
  rows_ = 0;
  clear_has_rows();
}
inline ::google::protobuf::int32 controlmsg::rows() const {
  // @@protoc_insertion_point(field_get:stradsccdmf.controlmsg.rows)
  return rows_;
}
inline void controlmsg::set_rows(::google::protobuf::int32 value) {
  set_has_rows();
  rows_ = value;
  // @@protoc_insertion_point(field_set:stradsccdmf.controlmsg.rows)
}

// optional int32 cols = 5;
inline bool controlmsg::has_cols() const {
  return (_has_bits_[0] & 0x00000010u) != 0;
}
inline void controlmsg::set_has_cols() {
  _has_bits_[0] |= 0x00000010u;
}
inline void controlmsg::clear_has_cols() {
  _has_bits_[0] &= ~0x00000010u;
}
inline void controlmsg::clear_cols() {
  cols_ = 0;
  clear_has_cols();
}
inline ::google::protobuf::int32 controlmsg::cols() const {
  // @@protoc_insertion_point(field_get:stradsccdmf.controlmsg.cols)
  return cols_;
}
inline void controlmsg::set_cols(::google::protobuf::int32 value) {
  set_has_cols();
  cols_ = value;
  // @@protoc_insertion_point(field_set:stradsccdmf.controlmsg.cols)
}

// repeated .stradsccdmf.initinfo init = 6;
inline int controlmsg::init_size() const {
  return init_.size();
}
inline void controlmsg::clear_init() {
  init_.Clear();
}
inline const ::stradsccdmf::initinfo& controlmsg::init(int index) const {
  // @@protoc_insertion_point(field_get:stradsccdmf.controlmsg.init)
  return init_.Get(index);
}
inline ::stradsccdmf::initinfo* controlmsg::mutable_init(int index) {
  // @@protoc_insertion_point(field_mutable:stradsccdmf.controlmsg.init)
  return init_.Mutable(index);
}
inline ::stradsccdmf::initinfo* controlmsg::add_init() {
  // @@protoc_insertion_point(field_add:stradsccdmf.controlmsg.init)
  return init_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::stradsccdmf::initinfo >&
controlmsg::init() const {
  // @@protoc_insertion_point(field_list:stradsccdmf.controlmsg.init)
  return init_;
}
inline ::google::protobuf::RepeatedPtrField< ::stradsccdmf::initinfo >*
controlmsg::mutable_init() {
  // @@protoc_insertion_point(field_mutable_list:stradsccdmf.controlmsg.init)
  return &init_;
}

// repeated .stradsccdmf.singlebucket buckets = 7;
inline int controlmsg::buckets_size() const {
  return buckets_.size();
}
inline void controlmsg::clear_buckets() {
  buckets_.Clear();
}
inline const ::stradsccdmf::singlebucket& controlmsg::buckets(int index) const {
  // @@protoc_insertion_point(field_get:stradsccdmf.controlmsg.buckets)
  return buckets_.Get(index);
}
inline ::stradsccdmf::singlebucket* controlmsg::mutable_buckets(int index) {
  // @@protoc_insertion_point(field_mutable:stradsccdmf.controlmsg.buckets)
  return buckets_.Mutable(index);
}
inline ::stradsccdmf::singlebucket* controlmsg::add_buckets() {
  // @@protoc_insertion_point(field_add:stradsccdmf.controlmsg.buckets)
  return buckets_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::stradsccdmf::singlebucket >&
controlmsg::buckets() const {
  // @@protoc_insertion_point(field_list:stradsccdmf.controlmsg.buckets)
  return buckets_;
}
inline ::google::protobuf::RepeatedPtrField< ::stradsccdmf::singlebucket >*
controlmsg::mutable_buckets() {
  // @@protoc_insertion_point(field_mutable_list:stradsccdmf.controlmsg.buckets)
  return &buckets_;
}

// optional double partialobj = 8;
inline bool controlmsg::has_partialobj() const {
  return (_has_bits_[0] & 0x00000080u) != 0;
}
inline void controlmsg::set_has_partialobj() {
  _has_bits_[0] |= 0x00000080u;
}
inline void controlmsg::clear_has_partialobj() {
  _has_bits_[0] &= ~0x00000080u;
}
inline void controlmsg::clear_partialobj() {
  partialobj_ = 0;
  clear_has_partialobj();
}
inline double controlmsg::partialobj() const {
  // @@protoc_insertion_point(field_get:stradsccdmf.controlmsg.partialobj)
  return partialobj_;
}
inline void controlmsg::set_partialobj(double value) {
  set_has_partialobj();
  partialobj_ = value;
  // @@protoc_insertion_point(field_set:stradsccdmf.controlmsg.partialobj)
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace stradsccdmf

#ifndef SWIG
namespace google {
namespace protobuf {

template <> struct is_proto_enum< ::stradsccdmf::controlmsg_msgtype> : ::google::protobuf::internal::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::stradsccdmf::controlmsg_msgtype>() {
  return ::stradsccdmf::controlmsg_msgtype_descriptor();
}

}  // namespace google
}  // namespace protobuf
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_ccdmf_2eproto__INCLUDED