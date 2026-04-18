#include "symbol_extraction.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <set>
#include <cctype>
#include "ast_parser.hpp"

namespace fs = std::filesystem;

namespace ast_distance {

const char* language_name(Language lang) {
    switch (lang) {
        case Language::RUST: return "Rust";
        case Language::KOTLIN: return "Kotlin";
        case Language::CPP: return "C++";
        case Language::PYTHON: return "Python";
        case Language::TYPESCRIPT: return "TypeScript";
        default: return "Unknown";
    }
}

// ============================================================================
// Name conversion utilities
// ============================================================================

std::string snake_to_camel(const std::string& snake) {
    std::string result;
    bool next_upper = false;
    for (char c : snake) {
        if (c == '_') {
            next_upper = true;
        } else {
            if (next_upper) {
                result += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                next_upper = false;
            } else {
                result += c;
            }
        }
    }
    return result;
}

std::string rust_qualified_to_kotlin(const std::string& qualified) {
    // "Foo::bar" -> "Foo.bar"
    std::string result = qualified;
    size_t pos = 0;
    while ((pos = result.find("::", pos)) != std::string::npos) {
        result.replace(pos, 2, ".");
        pos += 1;
    }
    return result;
}

namespace {

// ============================================================================
// File utilities
// ============================================================================

bool should_skip_path(const std::string& path) {
    return path.find("/test") != std::string::npos ||
           path.find("/build/") != std::string::npos ||
           path.find("/CMakeFiles/") != std::string::npos ||
           path.find("/cmake-build") != std::string::npos ||
           path.find("/target/") != std::string::npos ||
           path.find("/_deps/") != std::string::npos ||
           path.find("/node_modules/") != std::string::npos;
}

std::string read_file_content(const fs::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) return {};
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::string get_node_text(TSNode node, const std::string& source) {
    uint32_t start = ts_node_start_byte(node);
    uint32_t end = ts_node_end_byte(node);
    if (end > start && end <= source.length()) {
        return source.substr(start, end - start);
    }
    return {};
}

int get_node_line(TSNode node) {
    return static_cast<int>(ts_node_start_point(node).row) + 1;
}

// ============================================================================
// Rust symbol extraction
// ============================================================================

Visibility rust_extract_visibility(TSNode node) {
    uint32_t count = ts_node_child_count(node);
    for (uint32_t i = 0; i < count; ++i) {
        TSNode child = ts_node_child(node, i);
        std::string type(ts_node_type(child));
        if (type == "visibility_modifier") {
            if (ts_node_child_count(child) > 1) {
                return Visibility::CRATE;
            }
            return Visibility::PUBLIC;
        }
    }
    return Visibility::PRIVATE;
}

std::string rust_extract_name(TSNode node, const std::string& source, const char* name_type) {
    uint32_t count = ts_node_named_child_count(node);
    for (uint32_t i = 0; i < count; ++i) {
        TSNode child = ts_node_named_child(node, i);
        std::string type(ts_node_type(child));
        if (type == name_type) {
            return get_node_text(child, source);
        }
    }
    return {};
}

void rust_extract_symbols_recursive(
    TSNode node,
    const std::string& source,
    const std::string& file,
    const std::string& parent_type,
    SymbolTable& table)
{
    std::string type(ts_node_type(node));

    if (type == "function_item") {
        std::string name = rust_extract_name(node, source, "identifier");
        if (!name.empty()) {
            Symbol sym;
            sym.name = name;
            sym.kind = SymbolKind::FUNCTION;
            sym.visibility = rust_extract_visibility(node);
            sym.file = file;
            sym.line = get_node_line(node);
            if (!parent_type.empty()) {
                sym.kind = SymbolKind::IMPL_METHOD;
                sym.parent = parent_type;
                sym.qualified_name = parent_type + "::" + name;
            } else {
                sym.qualified_name = name;
            }
            table.add(std::move(sym));
        }
    } else if (type == "struct_item" || type == "enum_item") {
        std::string name = rust_extract_name(node, source, "type_identifier");
        if (!name.empty()) {
            Symbol sym;
            sym.name = name;
            sym.qualified_name = name;
            sym.kind = (type == "enum_item") ? SymbolKind::ENUM : SymbolKind::STRUCT;
            sym.file = file;
            sym.line = get_node_line(node);
            table.add(std::move(sym));
        }
    } else {
        uint32_t count = ts_node_named_child_count(node);
        for (uint32_t i = 0; i < count; ++i) {
            rust_extract_symbols_recursive(ts_node_named_child(node, i), source, file, parent_type, table);
        }
    }
}

// ============================================================================
// Kotlin symbol extraction
// ============================================================================

std::string kotlin_extract_name(TSNode node, const std::string& source) {
    uint32_t count = ts_node_named_child_count(node);
    for (uint32_t i = 0; i < count; ++i) {
        TSNode child = ts_node_named_child(node, i);
        if (std::string(ts_node_type(child)) == "simple_identifier") {
            return get_node_text(child, source);
        }
    }
    return {};
}

void kotlin_extract_symbols_recursive(
    TSNode node,
    const std::string& source,
    const std::string& file,
    const std::string& parent_type,
    SymbolTable& table)
{
    std::string type(ts_node_type(node));

    if (type == "function_declaration") {
        std::string name = kotlin_extract_name(node, source);
        if (!name.empty()) {
            Symbol sym;
            sym.name = name;
            sym.file = file;
            sym.line = get_node_line(node);
            if (!parent_type.empty()) {
                sym.kind = SymbolKind::IMPL_METHOD;
                sym.qualified_name = parent_type + "." + name;
            } else {
                sym.kind = SymbolKind::FUNCTION;
                sym.qualified_name = name;
            }
            table.add(std::move(sym));
        }
    } else if (type == "class_declaration" || type == "interface_declaration") {
        std::string name = kotlin_extract_name(node, source);
        if (!name.empty()) {
            Symbol sym;
            sym.name = name;
            sym.qualified_name = name;
            sym.kind = (type == "interface_declaration") ? SymbolKind::TRAIT : SymbolKind::STRUCT;
            sym.file = file;
            sym.line = get_node_line(node);
            
            TSNode body = ts_node_child_by_field_name(node, "body", 4);
            if (!ts_node_is_null(body)) {
                uint32_t count = ts_node_named_child_count(body);
                for (uint32_t i = 0; i < count; ++i) {
                    kotlin_extract_symbols_recursive(ts_node_named_child(body, i), source, file, name, table);
                }
            }
            table.add(std::move(sym));
        }
    } else {
        uint32_t count = ts_node_named_child_count(node);
        for (uint32_t i = 0; i < count; ++i) {
            kotlin_extract_symbols_recursive(ts_node_named_child(node, i), source, file, parent_type, table);
        }
    }
}

// ============================================================================
// Python symbol extraction
// ============================================================================

void python_extract_symbols_recursive(
    TSNode node,
    const std::string& source,
    const std::string& file,
    const std::string& parent_type,
    SymbolTable& table)
{
    std::string type(ts_node_type(node));

    if (type == "function_definition") {
        TSNode name_node = ts_node_child_by_field_name(node, "name", 4);
        if (!ts_node_is_null(name_node)) {
            std::string name = get_node_text(name_node, source);
            Symbol sym;
            sym.name = name;
            sym.visibility = name.starts_with("_") ? Visibility::PRIVATE : Visibility::PUBLIC;
            sym.file = file;
            sym.line = get_node_line(node);
            sym.parent = parent_type;
            if (!parent_type.empty()) {
                sym.kind = SymbolKind::IMPL_METHOD;
                sym.qualified_name = parent_type + "." + name;
            } else {
                sym.kind = SymbolKind::FUNCTION;
                sym.qualified_name = name;
            }
            table.add(std::move(sym));
        }
    } else if (type == "class_definition") {
        TSNode name_node = ts_node_child_by_field_name(node, "name", 4);
        if (!ts_node_is_null(name_node)) {
            std::string name = get_node_text(name_node, source);
            Symbol sym;
            sym.name = name;
            sym.qualified_name = name;
            sym.kind = SymbolKind::STRUCT;
            sym.visibility = name.starts_with("_") ? Visibility::PRIVATE : Visibility::PUBLIC;
            sym.file = file;
            sym.line = get_node_line(node);
            
            TSNode body = ts_node_child_by_field_name(node, "body", 4);
            if (!ts_node_is_null(body)) {
                uint32_t count = ts_node_named_child_count(body);
                for (uint32_t i = 0; i < count; ++i) {
                    TSNode member = ts_node_named_child(body, i);
                    python_extract_symbols_recursive(member, source, file, name, table);
                }
            }
            table.add(std::move(sym));
        }
    } else {
        uint32_t count = ts_node_named_child_count(node);
        for (uint32_t i = 0; i < count; ++i) {
            python_extract_symbols_recursive(ts_node_named_child(node, i), source, file, parent_type, table);
        }
    }
}

// ============================================================================
// TypeScript symbol extraction
// ============================================================================

void typescript_extract_symbols_recursive(
    TSNode node,
    const std::string& source,
    const std::string& file,
    const std::string& parent_type,
    SymbolTable& table)
{
    std::string type(ts_node_type(node));

    if (type == "function_declaration" || type == "method_definition" || type == "function_expression" || type == "arrow_function") {
        std::string name;
        TSNode name_node = ts_node_child_by_field_name(node, "name", 4);
        if (!ts_node_is_null(name_node)) {
            name = get_node_text(name_node, source);
        } else if (type == "arrow_function" || type == "function_expression") {
            TSNode parent = ts_node_parent(node);
            if (!ts_node_is_null(parent)) {
                std::string pt(ts_node_type(parent));
                if (pt == "variable_declarator" || pt == "property_definition") {
                    TSNode pname_node = ts_node_child_by_field_name(parent, "name", 4);
                    if (!ts_node_is_null(pname_node)) {
                        name = get_node_text(pname_node, source);
                    }
                }
            }
        }

        if (!name.empty()) {
            Symbol sym;
            sym.name = name;
            sym.file = file;
            sym.line = get_node_line(node);
            sym.parent = parent_type;
            if (name == "constructor") {
                sym.name = "__init__"; // Canonicalize
            }
            if (!parent_type.empty()) {
                sym.kind = SymbolKind::IMPL_METHOD;
                sym.qualified_name = parent_type + "." + sym.name;
            } else {
                sym.kind = SymbolKind::FUNCTION;
                sym.qualified_name = sym.name;
            }
            
            ASTParser ast_parser;
            sym.is_stub = ast_parser.has_stub_markers_in_node(node, source, Language::TYPESCRIPT);
            table.add(std::move(sym));
        }
    } else if (type == "class_declaration" || type == "interface_declaration") {
        TSNode name_node = ts_node_child_by_field_name(node, "name", 4);
        if (!ts_node_is_null(name_node)) {
            std::string name = get_node_text(name_node, source);
            Symbol sym;
            sym.name = name;
            sym.qualified_name = name;
            sym.kind = (type == "interface_declaration") ? SymbolKind::TRAIT : SymbolKind::STRUCT;
            sym.file = file;
            sym.line = get_node_line(node);
            
            TSNode body = ts_node_child_by_field_name(node, "body", 4);
            if (!ts_node_is_null(body)) {
                uint32_t count = ts_node_named_child_count(body);
                for (uint32_t i = 0; i < count; ++i) {
                    typescript_extract_symbols_recursive(ts_node_named_child(body, i), source, file, name, table);
                }
            }
            table.add(std::move(sym));
        }
    } else if (type == "enum_declaration") {
        TSNode name_node = ts_node_child_by_field_name(node, "name", 4);
        if (!ts_node_is_null(name_node)) {
            std::string name = get_node_text(name_node, source);
            Symbol sym;
            sym.name = name;
            sym.qualified_name = name;
            sym.kind = SymbolKind::ENUM;
            sym.file = file;
            sym.line = get_node_line(node);
            table.add(std::move(sym));
        }
    } else {
        uint32_t count = ts_node_named_child_count(node);
        for (uint32_t i = 0; i < count; ++i) {
            typescript_extract_symbols_recursive(ts_node_named_child(node, i), source, file, parent_type, table);
        }
    }
}

} // anonymous namespace

// ============================================================================
// Main Symbol Extraction Functions
// ============================================================================

SymbolTable extract_symbols(const std::string& root, Language lang) {
    switch (lang) {
        case Language::RUST: return extract_rust_symbols(root);
        case Language::KOTLIN: return extract_kotlin_symbols(root);
        case Language::PYTHON: return extract_python_symbols(root);
        case Language::TYPESCRIPT: return extract_typescript_symbols(root);
        default: return {};
    }
}

SymbolTable extract_rust_symbols(const std::string& root) {
    SymbolTable table;
    TSParser* parser = ts_parser_new();
    ts_parser_set_language(parser, tree_sitter_rust());

    if (!fs::exists(root)) return table;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        std::string path = entry.path().string();
        if (should_skip_path(path) || !path.ends_with(".rs")) continue;

        std::string content = read_file_content(entry.path());
        if (content.empty()) continue;
        std::string rel_path = fs::relative(entry.path(), root).string();

        TSTree* tree = ts_parser_parse_string(parser, nullptr, content.c_str(), content.length());
        if (!tree) continue;
        rust_extract_symbols_recursive(ts_tree_root_node(tree), content, rel_path, "", table);
        ts_tree_delete(tree);
    }
    ts_parser_delete(parser);
    return table;
}

SymbolTable extract_kotlin_symbols(const std::string& root) {
    SymbolTable table;
    TSParser* parser = ts_parser_new();
    ts_parser_set_language(parser, tree_sitter_kotlin());

    if (!fs::exists(root)) return table;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        std::string path = entry.path().string();
        if (should_skip_path(path) || !path.ends_with(".kt")) continue;

        std::string content = read_file_content(entry.path());
        if (content.empty()) continue;
        std::string rel_path = fs::relative(entry.path(), root).string();

        TSTree* tree = ts_parser_parse_string(parser, nullptr, content.c_str(), content.length());
        if (!tree) continue;
        kotlin_extract_symbols_recursive(ts_tree_root_node(tree), content, rel_path, "", table);
        ts_tree_delete(tree);
    }
    ts_parser_delete(parser);
    return table;
}

SymbolTable extract_python_symbols(const std::string& root) {
    SymbolTable table;
    TSParser* parser = ts_parser_new();
    ts_parser_set_language(parser, tree_sitter_python());

    if (!fs::exists(root)) return table;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        std::string path = entry.path().string();
        if (should_skip_path(path) || !path.ends_with(".py")) continue;

        std::string content = read_file_content(entry.path());
        if (content.empty()) continue;
        std::string rel_path = fs::relative(entry.path(), root).string();

        TSTree* tree = ts_parser_parse_string(parser, nullptr, content.c_str(), content.length());
        if (!tree) continue;
        python_extract_symbols_recursive(ts_tree_root_node(tree), content, rel_path, "", table);
        ts_tree_delete(tree);
    }
    ts_parser_delete(parser);
    return table;
}

SymbolTable extract_typescript_symbols(const std::string& root) {
    SymbolTable table;
    TSParser* parser = ts_parser_new();
    ts_parser_set_language(parser, tree_sitter_typescript());

    if (!fs::exists(root)) return table;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        std::string path = entry.path().string();
        if (should_skip_path(path) || (!path.ends_with(".ts") && !path.ends_with(".tsx"))) continue;

        std::string content = read_file_content(entry.path());
        if (content.empty()) continue;
        std::string rel_path = fs::relative(entry.path(), root).string();

        TSTree* tree = ts_parser_parse_string(parser, nullptr, content.c_str(), content.length());
        if (!tree) continue;
        typescript_extract_symbols_recursive(ts_tree_root_node(tree), content, rel_path, "", table);
        ts_tree_delete(tree);
    }
    ts_parser_delete(parser);
    return table;
}

// ============================================================================
// Symbol matching
// ============================================================================

struct MatchCandidate {
    std::string target_name;
    float confidence;
    std::string reason;
};

std::vector<MatchCandidate> generate_match_candidates(const Symbol& src_sym, Language src_lang, Language tgt_lang) {
    std::vector<MatchCandidate> candidates;
    candidates.push_back({src_sym.name, 1.0f, "exact"});

    // snake_case -> camelCase
    std::string camel = snake_to_camel(src_sym.name);
    if (camel != src_sym.name) {
        candidates.push_back({camel, 0.95f, "camelCase"});
    }

    if (!src_sym.parent.empty()) {
        std::string qual = src_sym.parent + "." + src_sym.name;
        candidates.push_back({qual, 0.9f, "qualified"});
        
        std::string qual_camel = src_sym.parent + "." + camel;
        if (qual_camel != qual) {
            candidates.push_back({qual_camel, 0.85f, "qualified+camelCase"});
        }
    }

    return candidates;
}

SymbolParityReport build_parity_report(const SymbolTable& source, const SymbolTable& target, Language src_lang, Language tgt_lang) {
    SymbolParityReport report;

    for (const auto& sym : target.symbols) {
        if (sym.is_stub) report.stub_count++;
    }

    std::map<std::string, const Symbol*> tgt_by_name;
    std::map<std::string, const Symbol*> tgt_by_qualified;
    for (const auto& sym : target.symbols) {
        tgt_by_name[sym.name] = &sym;
        tgt_by_qualified[sym.qualified_name] = &sym;
    }

    std::set<const Symbol*> matched_target;

    for (const auto& src_sym : source.symbols) {
        auto candidates = generate_match_candidates(src_sym, src_lang, tgt_lang);
        const Symbol* best_match = nullptr;
        float best_confidence = 0.0f;
        std::string best_reason;

        for (const auto& candidate : candidates) {
            auto it = tgt_by_qualified.find(candidate.target_name);
            if (it != tgt_by_qualified.end()) {
                if (candidate.confidence > best_confidence) {
                    best_match = it->second;
                    best_confidence = candidate.confidence;
                    best_reason = candidate.reason;
                }
            }
            auto it2 = tgt_by_name.find(candidate.target_name);
            if (it2 != tgt_by_name.end()) {
                if (candidate.confidence > best_confidence) {
                    best_match = it2->second;
                    best_confidence = candidate.confidence;
                    best_reason = candidate.reason;
                }
            }
        }

        SymbolMatch match;
        match.source_symbol = &src_sym;
        match.target_symbol = best_match;
        match.confidence = best_confidence;
        match.match_reason = best_reason;
        report.matches.push_back(match);

        if (best_match) {
            matched_target.insert(best_match);
            if (src_sym.is_test) report.test_coverage[src_sym.kind].first++;
            else report.coverage[src_sym.kind].first++;
        } else {
            report.missing_in_target.push_back(&src_sym);
        }
        if (src_sym.is_test) report.test_coverage[src_sym.kind].second++;
        else report.coverage[src_sym.kind].second++;
    }

    for (const auto& tgt_sym : target.symbols) {
        if (!matched_target.count(&tgt_sym)) {
            report.extra_in_target.push_back(&tgt_sym);
        }
    }

    return report;
}

void SymbolParityReport::print(Language src_lang, Language tgt_lang, bool verbose, bool missing_only) const {
    std::cout << "==========================================================\n";
    std::cout << "SYMBOL PARITY: " << language_name(src_lang) << " -> " << language_name(tgt_lang) << "\n";
    std::cout << "==========================================================\n";

    std::vector<SymbolKind> primary_kinds = {
        SymbolKind::FUNCTION, SymbolKind::STRUCT, SymbolKind::ENUM,
        SymbolKind::TRAIT, SymbolKind::IMPL_METHOD
    };

    auto print_kind_detail = [&](SymbolKind kind, bool is_test) {
        for (const auto& m : matches) {
            if (m.source_symbol->kind != kind || m.source_symbol->is_test != is_test) continue;
            if (missing_only) {
                if (!m.target_symbol) {
                    std::cout << "  " << m.source_symbol->name << " (" << m.source_symbol->file << ":" << m.source_symbol->line << ")\n";
                }
            } else if (verbose) {
                if (m.target_symbol) {
                    std::cout << "  [MATCHED] " << m.source_symbol->name << " -> " << m.target_symbol->name << " (" << m.match_reason << ")\n";
                } else {
                    std::cout << "  [MISSING] " << m.source_symbol->name << " (" << m.source_symbol->file << ":" << m.source_symbol->line << ")\n";
                }
            } else if (!m.target_symbol) {
                std::cout << "  [MISSING] " << m.source_symbol->name << " (" << m.source_symbol->file << ":" << m.source_symbol->line << ")\n";
            }
        }
    };

    for (SymbolKind kind : primary_kinds) {
        auto it = coverage.find(kind);
        if (it == coverage.end() || it->second.second == 0) continue;
        int matched = it->second.first;
        int total = it->second.second;
        float pct = (total > 0) ? (100.0f * matched / total) : 0.0f;
        std::cout << "\n--- " << symbol_kind_label(kind) << " (" << matched << "/" << total << " matched, " << std::fixed << std::setprecision(1) << pct << "%) ---\n";
        print_kind_detail(kind, false);
    }
}

void SymbolParityReport::print_json() const {
    std::cout << "{\n  \"matches\": [\n";
    for (size_t i = 0; i < matches.size(); ++i) {
        const auto& m = matches[i];
        std::cout << "    {\"source\": \"" << m.source_symbol->qualified_name << "\", \"target\": " << (m.target_symbol ? "\"" + m.target_symbol->qualified_name + "\"" : "null") << "}";
        if (i < matches.size() - 1) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "  ]\n}\n";
}

void cmd_symbol_parity(const std::string& source_root,
                       const std::string& target_root,
                       const SymbolParityOptions& options) {
    std::cerr << "Extracting source symbols (" << language_name(options.source_lang) << ") from " << source_root << "...\n";
    SymbolTable source = extract_symbols(source_root, options.source_lang);
    std::cerr << "Found " << source.size() << " symbols\n";

    std::cerr << "Extracting target symbols (" << language_name(options.target_lang) << ") from " << target_root << "...\n";
    SymbolTable target = extract_symbols(target_root, options.target_lang);
    std::cerr << "Found " << target.size() << " symbols\n";

    SymbolParityReport report = build_parity_report(source, target, options.source_lang, options.target_lang);
    if (options.json) report.print_json();
    else report.print(options.source_lang, options.target_lang, options.verbose, options.missing_only);
}

void cmd_import_map(const std::string& kotlin_root, const ImportMapOptions& options) {
    std::cerr << "Warning: cmd_import_map not yet fully generalized\n";
}

void cmd_compiler_fixup(const std::string& kotlin_root, const std::string& error_file, const CompilerFixupOptions& options) {
    std::cerr << "Warning: cmd_compiler_fixup not yet fully generalized\n";
}

} // namespace ast_distance
