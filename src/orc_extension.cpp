#define DUCKDB_EXTENSION_MAIN

#include "orc_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/multi_file_reader.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include <orc/OrcFile.hh>

namespace duckdb {

struct ORCType {
    ORCType() : duckdb_type(LogicalType::INVALID) {}
    
    ORCType(orc::TypeKind orc_type_p, LogicalType duckdb_type_p) 
        : duckdb_type(std::move(duckdb_type_p)), orc_type(orc_type_p) {}

    LogicalType duckdb_type;
    orc::TypeKind orc_type;
    vector<pair<string, unique_ptr<ORCType>>> children;

    void AddChild(const string& name, unique_ptr<ORCType> child) {
        children.emplace_back(name, std::move(child));
    }
};

// Fix for TypeKind_Name undefined error
static string GetORCTypeName(orc::TypeKind type) {
    switch(type) {
        case orc::TypeKind::BOOLEAN: return "BOOLEAN";
        case orc::TypeKind::BYTE: return "BYTE";
        case orc::TypeKind::SHORT: return "SHORT";
        case orc::TypeKind::INT: return "INT";
        case orc::TypeKind::LONG: return "LONG";
        case orc::TypeKind::FLOAT: return "FLOAT";
        case orc::TypeKind::DOUBLE: return "DOUBLE";
        case orc::TypeKind::STRING: return "STRING";
        case orc::TypeKind::BINARY: return "BINARY";
        case orc::TypeKind::TIMESTAMP: return "TIMESTAMP";
        case orc::TypeKind::LIST: return "LIST";
        case orc::TypeKind::MAP: return "MAP";
        case orc::TypeKind::STRUCT: return "STRUCT";
        default: return "UNKNOWN";
    }
}

class ORCMemoryInputStream : public orc::InputStream {
public:
    ORCMemoryInputStream(const char* data, uint64_t length) 
        : data_(data), length_(length), position_(0) {}

    uint64_t getLength() const override { return length_; }
    uint64_t getNaturalReadSize() const override { return 128 * 1024; }

    void read(void* buf, uint64_t length, uint64_t offset) override {
        if (offset + length > length_) {
            throw std::runtime_error("Read past end of stream");
        }
        memcpy(buf, data_ + offset, length);
    }

    const char* getData() const { return data_; }

private:
    const char* data_;
    uint64_t length_;
    uint64_t position_;
};

struct ORCOptions {
    explicit ORCOptions() {}

    void Serialize(Serializer &serializer) const {
        file_options.Serialize(serializer);
    }

    static ORCOptions Deserialize(Deserializer &deserializer) {
        ORCOptions options;
        options.file_options = MultiFileReaderOptions::Deserialize(deserializer);
        return options;
    }

    MultiFileReaderOptions file_options;
};

static unique_ptr<ORCType> TransformSchema(const orc::Type* type) {
    switch (type->getKind()) {
    case orc::TypeKind::BOOLEAN:
        return make_unique<ORCType>(orc::TypeKind::BOOLEAN, LogicalType::BOOLEAN);
    case orc::TypeKind::BYTE:
        return make_unique<ORCType>(orc::TypeKind::BYTE, LogicalType::TINYINT);
    case orc::TypeKind::SHORT:
        return make_unique<ORCType>(orc::TypeKind::SHORT, LogicalType::SMALLINT);
    case orc::TypeKind::INT:
        return make_unique<ORCType>(orc::TypeKind::INT, LogicalType::INTEGER);
    case orc::TypeKind::LONG:
        return make_unique<ORCType>(orc::TypeKind::LONG, LogicalType::BIGINT);
    case orc::TypeKind::FLOAT:
        return make_unique<ORCType>(orc::TypeKind::FLOAT, LogicalType::FLOAT);
    case orc::TypeKind::DOUBLE:
        return make_unique<ORCType>(orc::TypeKind::DOUBLE, LogicalType::DOUBLE);
    case orc::TypeKind::STRING:
        return make_unique<ORCType>(orc::TypeKind::STRING, LogicalType::VARCHAR);
    case orc::TypeKind::BINARY:
        return make_unique<ORCType>(orc::TypeKind::BINARY, LogicalType::BLOB);
    case orc::TypeKind::TIMESTAMP:
        return make_unique<ORCType>(orc::TypeKind::TIMESTAMP, LogicalType::TIMESTAMP);
    case orc::TypeKind::LIST: {
        auto element_type = TransformSchema(type->getSubtype(0));
        auto list_type = make_unique<ORCType>(orc::TypeKind::LIST, 
                                            LogicalType::LIST(element_type->duckdb_type));
        list_type->AddChild("element", std::move(element_type));
        return list_type;
    }
    case orc::TypeKind::MAP: {
        auto key_type = TransformSchema(type->getSubtype(0));
        auto value_type = TransformSchema(type->getSubtype(1));
        auto map_type = make_unique<ORCType>(orc::TypeKind::MAP,
                                           LogicalType::MAP(key_type->duckdb_type, 
                                                          value_type->duckdb_type));
        map_type->AddChild("key", std::move(key_type));
        map_type->AddChild("value", std::move(value_type));
        return map_type;
    }
    case orc::TypeKind::STRUCT: {
        child_list_t<LogicalType> struct_children;
        auto struct_type = make_unique<ORCType>(orc::TypeKind::STRUCT, LogicalType::INVALID);
        
        for (size_t i = 0; i < type->getSubtypeCount(); i++) {
            auto field_name = type->getFieldName(i);
            auto field_type = TransformSchema(type->getSubtype(i));
            struct_children.push_back(make_pair(field_name, field_type->duckdb_type));
            struct_type->AddChild(field_name, std::move(field_type));
        }
        struct_type->duckdb_type = LogicalType::STRUCT(struct_children);
        return struct_type;
    }
    default:
        throw NotImplementedException("Unsupported ORC type: %s",
                                    GetORCTypeName(type->getKind()).c_str());
    }
}

static void TransformValue(const orc::ColumnVectorBatch* batch, 
                         const ORCType &orc_type,
                         Vector &target, idx_t row_idx) {
    if (batch->hasNulls && !batch->notNull[row_idx]) {
        FlatVector::SetNull(target, row_idx, true);
        return;
    }

    switch (orc_type.duckdb_type.id()) {
    case LogicalTypeId::BOOLEAN: {
        auto* long_batch = dynamic_cast<const orc::LongVectorBatch*>(batch);
        FlatVector::GetData<bool>(target)[row_idx] = long_batch->data[row_idx];
        break;
    }
    case LogicalTypeId::TINYINT: {
        auto* long_batch = dynamic_cast<const orc::LongVectorBatch*>(batch);
        FlatVector::GetData<int8_t>(target)[row_idx] = long_batch->data[row_idx];
        break;
    }
    case LogicalTypeId::SMALLINT: {
        auto* long_batch = dynamic_cast<const orc::LongVectorBatch*>(batch);
        FlatVector::GetData<int16_t>(target)[row_idx] = long_batch->data[row_idx];
        break;
    }
    case LogicalTypeId::INTEGER: {
        auto* long_batch = dynamic_cast<const orc::LongVectorBatch*>(batch);
        FlatVector::GetData<int32_t>(target)[row_idx] = long_batch->data[row_idx];
        break;
    }
    case LogicalTypeId::BIGINT: {
        auto* long_batch = dynamic_cast<const orc::LongVectorBatch*>(batch);
        FlatVector::GetData<int64_t>(target)[row_idx] = long_batch->data[row_idx];
        break;
    }
    case LogicalTypeId::FLOAT: {
        auto* double_batch = dynamic_cast<const orc::DoubleVectorBatch*>(batch);
        FlatVector::GetData<float>(target)[row_idx] = double_batch->data[row_idx];
        break;
    }
    case LogicalTypeId::DOUBLE: {
        auto* double_batch = dynamic_cast<const orc::DoubleVectorBatch*>(batch);
        FlatVector::GetData<double>(target)[row_idx] = double_batch->data[row_idx];
        break;
    }
    case LogicalTypeId::VARCHAR: {
        auto* string_batch = dynamic_cast<const orc::StringVectorBatch*>(batch);
        auto str_ptr = string_batch->data[row_idx];
        auto str_len = string_batch->length[row_idx];
        FlatVector::GetData<string_t>(target)[row_idx] = 
            StringVector::AddString(target, str_ptr, str_len);
        break;
    }
    case LogicalTypeId::BLOB: {
        auto* binary_batch = dynamic_cast<const orc::StringVectorBatch*>(batch);
        auto data_ptr = binary_batch->data[row_idx];
        auto data_len = binary_batch->length[row_idx];
        FlatVector::GetData<string_t>(target)[row_idx] = 
            StringVector::AddStringOrBlob(target, data_ptr, data_len);
        break;
    }
    case LogicalTypeId::TIMESTAMP: {
        auto* ts_batch = dynamic_cast<const orc::TimestampVectorBatch*>(batch);
        auto seconds = ts_batch->data[row_idx];
        auto nanos = ts_batch->nanoseconds[row_idx];
        timestamp_t ts(seconds * Interval::MICROS_PER_SEC + nanos / Interval::NANOS_PER_MICRO);
        FlatVector::GetData<timestamp_t>(target)[row_idx] = ts;
        break;
    }
    case LogicalTypeId::STRUCT: {
        auto* struct_batch = dynamic_cast<const orc::StructVectorBatch*>(batch);
        for (idx_t i = 0; i < orc_type.children.size(); i++) {
            TransformValue(struct_batch->fields[i], *orc_type.children[i].second,
                         *StructVector::GetEntries(target)[i], row_idx);
        }
        break;
    }
    case LogicalTypeId::LIST: {
        auto* list_batch = dynamic_cast<const orc::ListVectorBatch*>(batch);
        auto &entry_vector = ListVector::GetEntry(target);
        auto offsets = list_batch->offsets[row_idx];
        auto length = list_batch->offsets[row_idx + 1] - offsets;
        
        for (idx_t i = 0; i < length; i++) {
            TransformValue(list_batch->elements.get(), *orc_type.children[0].second,
                         entry_vector, offsets + i);
        }
        
        auto list_data = ListVector::GetData(target);
        list_data[row_idx].offset = offsets;
        list_data[row_idx].length = length;
        break;
    }
    case LogicalTypeId::MAP: {
        auto* map_batch = dynamic_cast<const orc::MapVectorBatch*>(batch);
        auto &key_vector = MapVector::GetKeys(target);
        auto &value_vector = MapVector::GetValues(target);
        auto offsets = map_batch->offsets[row_idx];
        auto length = map_batch->offsets[row_idx + 1] - offsets;

        for (idx_t i = 0; i < length; i++) {
            TransformValue(map_batch->keys.get(), *orc_type.children[0].second,
                         key_vector, offsets + i);
            TransformValue(map_batch->elements.get(), *orc_type.children[1].second,
                         value_vector, offsets + i);
        }

        auto map_data = ListVector::GetData(target);
        map_data[row_idx].offset = offsets;
        map_data[row_idx].length = length;
        break;
    }
    default:
        throw NotImplementedException("Unsupported type for ORC conversion: %s",
                                    orc_type.duckdb_type.ToString());
    }
}

struct ORCReader {
    ~ORCReader() {
        batch.reset();
        row_reader.reset();
        reader.reset();
    }

    void Read(DataChunk &output, const vector<column_t> &column_ids) {
        idx_t output_idx = 0;

        while (output_idx < STANDARD_VECTOR_SIZE) {
            if (!batch || current_row >= batch->numElements) {
                if (!row_reader->next(*batch)) {
                    break;
                }
                current_row = 0;
            }

            TransformValue(batch.get(), orc_type, *read_vec, current_row);
            current_row++;
            output_idx++;
        }

        if (duckdb_type.id() == LogicalTypeId::STRUCT) {
            for (idx_t col_idx = 0; col_idx < column_ids.size(); col_idx++) {
                if (column_ids[col_idx] >= names.size()) {
                    continue;
                }
                output.data[col_idx].Reference(
                    *StructVector::GetEntries(*read_vec)[column_ids[col_idx]]);
            }
        } else {
            output.data[column_ids[0]].Reference(*read_vec);
        }

        output.SetCardinality(output_idx);
    }

    ORCReader(ClientContext &context, const string& filename_p,
          const ORCOptions& options_p) {
    	filename = filename_p;
    	options = options_p;

    	auto &fs = FileSystem::GetFileSystem(context);
    	if (!fs.FileExists(filename)) {
    	    throw InvalidInputException("ORC file %s not found", filename);
    	}

	    auto file = fs.OpenFile(filename_p, FileFlags::FILE_FLAGS_READ);
	    auto file_size = file->GetFileSize();
	    allocated_data = Allocator::Get(context).Allocate(file_size);
	    auto bytes_read = file->Read(allocated_data.get(), file_size);

	    if (bytes_read != file_size) {
	        throw IOException("Failed to read entire file");
	    }

	    auto input_stream = unique_ptr<ORCMemoryInputStream>(
	        new ORCMemoryInputStream(const_char_ptr_cast(allocated_data.get()), file_size));
	    reader = unique_ptr<orc::Reader>(
	        orc::createReader(std::move(input_stream), orc::ReaderOptions()));
    
	    row_reader = unique_ptr<orc::RowReader>(
	        reader->createRowReader(orc::RowReaderOptions()));
	    batch = row_reader->createRowBatch(STANDARD_VECTOR_SIZE);

	    auto schema = reader->getType();
	    auto orc_schema = TransformSchema(schema.get());
	    duckdb_type = orc_schema->duckdb_type;
	    orc_type = std::move(*orc_schema);
	    read_vec = make_uniq<Vector>(duckdb_type);
	    current_row = batch->numElements; // Force first read

	    if (duckdb_type.id() == LogicalTypeId::STRUCT) {
	        for (idx_t child_idx = 0; child_idx < StructType::GetChildCount(duckdb_type); child_idx++) {
	            names.push_back(StructType::GetChildName(duckdb_type, child_idx));
	            return_types.push_back(StructType::GetChildType(duckdb_type, child_idx));
	        }
	    } else {
	        names.push_back("orc_data");
	        return_types.push_back(duckdb_type);
	    }
	}

    const string &GetFileName() { return filename; }
    const vector<string> &GetNames() { return names; }
    const vector<LogicalType> &GetTypes() { return return_types; }

    unique_ptr<orc::Reader> reader;
    unique_ptr<orc::RowReader> row_reader;
    unique_ptr<orc::ColumnVectorBatch> batch;
    AllocatedData allocated_data;
    unique_ptr<Vector> read_vec;
    ORCType orc_type;
    LogicalType duckdb_type;
    vector<LogicalType> return_types;
    vector<string> names;
    ORCOptions options;
    MultiFileReaderData reader_data;
    string filename;
    idx_t current_row;
};

struct ORCBindData : public TableFunctionData {
    shared_ptr<MultiFileList> file_list;
    unique_ptr<MultiFileReader> multi_file_reader;
    MultiFileReaderBindData reader_bind;
    vector<string> names;
    vector<LogicalType> types;
    ORCOptions orc_options;
    shared_ptr<ORCReader> initial_reader;

    void Initialize(shared_ptr<ORCReader> reader) {
        initial_reader = std::move(reader);
        orc_options = initial_reader->options;
    }

    void Initialize(ClientContext &, shared_ptr<ORCReader> reader) {
        Initialize(reader);
    }
};

struct ORCGlobalState : public GlobalTableFunctionState {
    mutex lock;
    MultiFileListScanData scan_data;
    shared_ptr<ORCReader> reader;
    vector<column_t> column_ids;
    optional_ptr<TableFilterSet> filters;
};

static bool ORCNextFile(ClientContext &context, const ORCBindData &bind_data,
                       ORCGlobalState &global_state,
                       shared_ptr<ORCReader> initial_reader) {
    unique_lock<mutex> parallel_lock(global_state.lock);

    string file;
    if (!bind_data.file_list->Scan(global_state.scan_data, file)) {
        return false;
    }

    if (initial_reader) {
        D_ASSERT(file == initial_reader->GetFileName());
        global_state.reader = initial_reader;
    } else {
        global_state.reader = make_shared<ORCReader>(context, file, bind_data.orc_options);
    }

    bind_data.multi_file_reader->InitializeReader(
        *global_state.reader, bind_data.orc_options.file_options,
        bind_data.reader_bind, bind_data.types, bind_data.names,
        global_state.column_ids, global_state.filters, file, context, nullptr);
    return true;
}

static unique_ptr<FunctionData> ORCBindFunction(ClientContext &context, 
                                              TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types, 
                                              vector<string> &names) {
    auto result = make_uniq<ORCBindData>();
    result->multi_file_reader = MultiFileReader::Create(input.table_function);

    for (auto &kv : input.named_parameters) {
        if (kv.second.IsNull()) {
            throw BinderException("Cannot use NULL as function argument");
        }
        auto loption = StringUtil::Lower(kv.first);
        if (result->multi_file_reader->ParseOption(
                kv.first, kv.second, result->orc_options.file_options, context)) {
            continue;
        }
        throw InternalException("Unrecognized option %s", loption.c_str());
    }

    result->file_list = result->multi_file_reader->CreateFileList(context, input.inputs[0]);
    result->reader_bind = result->multi_file_reader->BindReader<ORCReader>(
        context, result->types, result->names, *result->file_list, *result,
        result->orc_options);

    return_types = result->types;
    names = result->names;

    return result;
}

static unique_ptr<GlobalTableFunctionState> ORCGlobalInit(ClientContext &context,
                                                        TableFunctionInitInput &input) {
    auto result = make_uniq<ORCGlobalState>();
    auto &bind_data = input.bind_data->Cast<ORCBindData>();

    result->column_ids = input.column_ids;
    result->filters = input.filters;

    bind_data.file_list->InitializeScan(result->scan_data);
    if (!ORCNextFile(context, bind_data, *result, bind_data.initial_reader)) {
        throw InternalException("Cannot scan ORC files");
    }
    return result;
}

static void ORCTableFunction(ClientContext &context, TableFunctionInput &data,
                           DataChunk &output) {
    auto &bind_data = data.bind_data->Cast<ORCBindData>();
    auto &global_state = data.global_state->Cast<ORCGlobalState>();
    
    do {
        output.Reset();
        global_state.reader->Read(output, global_state.column_ids);
        bind_data.multi_file_reader->FinalizeChunk(context, bind_data.reader_bind,
                                                 global_state.reader->reader_data,
                                                 output, nullptr);
        if (output.size() > 0) {
            return;
        }
        if (!ORCNextFile(context, bind_data, global_state, nullptr)) {
            return;
        }
    } while (true);
}

static void LoadInternal(DatabaseInstance &instance) {
    auto table_function = TableFunction("read_orc", {LogicalType::VARCHAR},
                                      ORCTableFunction, ORCBindFunction, ORCGlobalInit);
    table_function.projection_pushdown = true;
    MultiFileReader::AddParameters(table_function);
    ExtensionUtil::RegisterFunction(instance,
                                  MultiFileReader::CreateFunctionSet(table_function));
}

void OrcExtension::Load(DuckDB &db) { LoadInternal(*db.instance); }
std::string OrcExtension::Name() { return "orc"; }
std::string OrcExtension::Version() const { return "0.0.1"; }

} // namespace duckdb

extern "C" {
DUCKDB_EXTENSION_API void orc_init(duckdb::DatabaseInstance &db) {
    duckdb::DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<duckdb::OrcExtension>();
}

DUCKDB_EXTENSION_API const char *orc_version() {
    return duckdb::DuckDB::LibraryVersion();
}
}
