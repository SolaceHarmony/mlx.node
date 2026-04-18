"""Normalized AST node types that map across Rust, Kotlin, C++, and Python.

Faithful transliteration of include/node_types.hpp from the C++ ast_distance tool.
Based on the ASTERIA paper's categorization (Table I).
"""

from __future__ import annotations

from enum import IntEnum


class NodeType(IntEnum):
    """Normalized AST node types that map across languages.

    These types provide a common vocabulary for comparing ASTs from different
    languages (Rust, Kotlin, C++, Python) by mapping language-specific node types
    to a normalized set of semantic categories.
    """

    # Statement nodes (control flow)
    BLOCK = 0
    IF = 1
    FOR = 2
    WHILE = 3
    SWITCH = 4       # match in Rust, when in Kotlin
    RETURN = 5
    GOTO = 6         # break/continue with label
    CONTINUE = 7
    BREAK = 8
    TRY = 9
    THROW = 10

    # Expression nodes - assignments
    ASSIGN = 11
    ASSIGN_ADD = 12
    ASSIGN_SUB = 13
    ASSIGN_MUL = 14
    ASSIGN_DIV = 15
    ASSIGN_MOD = 16
    ASSIGN_AND = 17
    ASSIGN_OR = 18
    ASSIGN_XOR = 19

    # Expression nodes - comparisons
    EQ = 20
    NE = 21
    GT = 22
    LT = 23
    GE = 24
    LE = 25

    # Expression nodes - arithmetic
    ADD = 26
    SUB = 27
    MUL = 28
    DIV = 29
    MOD = 30
    NEG = 31

    # Expression nodes - bitwise
    BIT_AND = 32
    BIT_OR = 33
    BIT_XOR = 34
    BIT_NOT = 35
    SHL = 36
    SHR = 37

    # Expression nodes - logical
    AND = 38
    OR = 39
    NOT = 40

    # Expression nodes - other
    INDEX = 41
    FIELD_ACCESS = 42
    CALL = 43
    METHOD_CALL = 44
    LAMBDA = 45
    TERNARY = 46
    CAST = 47
    RANGE = 48

    # Literals and identifiers
    VARIABLE = 50
    NUMBER = 51
    STRING = 52
    BOOLEAN = 53
    NULL_LIT = 54
    CHAR = 55

    # Declarations
    FUNCTION = 60
    CLASS = 61
    STRUCT = 62
    ENUM = 63
    INTERFACE = 64   # trait in Rust
    VAR_DECL = 65
    PARAM = 66
    TYPE_PARAM = 67

    # Type annotations
    TYPE_REF = 70
    ARRAY_TYPE = 71
    NULLABLE_TYPE = 72
    FUNC_TYPE = 73
    GENERIC_TYPE = 74

    # Operators (captured from unnamed nodes for semantic awareness)
    ARITHMETIC_OP = 75
    COMPARISON_OP = 76
    LOGICAL_OP = 77
    BITWISE_OP = 78
    ASSIGNMENT_OP = 79

    # Other
    COMMENT = 80
    IMPORT = 81
    PACKAGE = 82
    ANNOTATION = 83
    MODIFIER = 84
    OTHER = 85

    # Unknown/unhandled
    UNKNOWN = 99

    NUM_TYPES = 100


# ── Rust AST node type mappings (from tree-sitter-rust) ──────────────────────
# Faithful transliteration of rust_node_to_type from node_types.hpp

_RUST_MAP: dict[str, NodeType] = {
    # Statements
    "block": NodeType.BLOCK,
    "if_expression": NodeType.IF,
    "if_let_expression": NodeType.IF,
    "for_expression": NodeType.FOR,
    "while_expression": NodeType.WHILE,
    "while_let_expression": NodeType.WHILE,
    "loop_expression": NodeType.WHILE,
    "match_expression": NodeType.SWITCH,
    "match_arm": NodeType.BLOCK,             # case arm → block-like
    "match_pattern": NodeType.OTHER,
    "return_expression": NodeType.RETURN,
    "continue_expression": NodeType.CONTINUE,
    "break_expression": NodeType.BREAK,
    "try_expression": NodeType.TRY,
    "expression_statement": NodeType.OTHER,   # expression used as statement
    "unsafe_block": NodeType.BLOCK,

    # Assignments
    "assignment_expression": NodeType.ASSIGN,
    "compound_assignment_expr": NodeType.ASSIGN,

    # Expressions
    "binary_expression": NodeType.OTHER,     # operator captured from unnamed children
    "unary_expression": NodeType.OTHER,
    "reference_expression": NodeType.OTHER,  # &foo, &mut foo
    "dereference_expression": NodeType.OTHER, # *foo
    "range_expression": NodeType.RANGE,
    "parenthesized_expression": NodeType.OTHER,
    "array_expression": NodeType.OTHER,       # array literal [a, b, c]
    "tuple_expression": NodeType.OTHER,
    "unit_expression": NodeType.OTHER,        # ()
    "type_cast_expression": NodeType.CAST,

    # Literals
    "identifier": NodeType.VARIABLE,
    "field_identifier": NodeType.VARIABLE,    # field name in field_expression
    "integer_literal": NodeType.NUMBER,
    "float_literal": NodeType.NUMBER,
    "string_literal": NodeType.STRING,
    "raw_string_literal": NodeType.STRING,
    "string_content": NodeType.STRING,        # inner text of string literals
    "boolean_literal": NodeType.BOOLEAN,
    "char_literal": NodeType.CHAR,
    "self": NodeType.VARIABLE,                # self keyword → variable
    "super": NodeType.OTHER,                  # super keyword in paths
    "crate": NodeType.OTHER,                  # crate keyword in paths
    "shorthand_field_identifier": NodeType.VARIABLE, # shorthand field in patterns

    # Function/Method calls
    "call_expression": NodeType.CALL,
    "method_call_expression": NodeType.METHOD_CALL,
    "macro_invocation": NodeType.CALL,        # macro!() → call-like
    "generic_function": NodeType.CALL,        # foo::<T>() → call-like
    "arguments": NodeType.OTHER,
    "parameters": NodeType.OTHER,
    "closure_parameters": NodeType.OTHER,     # |x, y| closure params

    # Declarations
    "source_file": NodeType.BLOCK,            # root module → block
    "function_item": NodeType.FUNCTION,
    "struct_item": NodeType.STRUCT,
    "enum_item": NodeType.ENUM,
    "enum_variant_list": NodeType.BLOCK,
    "enum_variant": NodeType.VAR_DECL,
    "trait_item": NodeType.INTERFACE,
    "impl_item": NodeType.BLOCK,              # Treat like block (matches Kotlin class_body → BLOCK)
    "declaration_list": NodeType.BLOCK,       # impl body — maps to Kotlin class_body (also BLOCK)
    "let_declaration": NodeType.VAR_DECL,
    "const_item": NodeType.VAR_DECL,
    "static_item": NodeType.VAR_DECL,
    "type_item": NodeType.VAR_DECL,           # type alias
    "function_signature_item": NodeType.FUNCTION, # trait method signature
    "parameter": NodeType.PARAM,
    "self_parameter": NodeType.PARAM,
    "type_parameter": NodeType.TYPE_PARAM,
    "constrained_type_parameter": NodeType.TYPE_PARAM,
    "field_declaration_list": NodeType.BLOCK,
    "ordered_field_declaration_list": NodeType.BLOCK, # tuple struct fields
    "field_declaration": NodeType.VAR_DECL,
    "struct_expression": NodeType.CALL,       # Struct { .. } → constructor call-like
    "field_initializer_list": NodeType.OTHER,
    "field_initializer": NodeType.ASSIGN,
    "shorthand_field_initializer": NodeType.ASSIGN, # Struct { field } shorthand
    "base_field_initializer": NodeType.ASSIGN,  # ..base
    "mod_item": NodeType.PACKAGE,

    # Field/Index access
    "field_expression": NodeType.FIELD_ACCESS,
    "index_expression": NodeType.INDEX,

    # Closure
    "closure_expression": NodeType.LAMBDA,

    # Types
    "type_identifier": NodeType.TYPE_REF,
    "primitive_type": NodeType.TYPE_REF,      # i32, bool, str etc
    "scoped_identifier": NodeType.FIELD_ACCESS, # path::to::Type → field access chain
    "scoped_type_identifier": NodeType.TYPE_REF,
    "array_type": NodeType.ARRAY_TYPE,
    "generic_type": NodeType.GENERIC_TYPE,
    "generic_type_with_turbofish": NodeType.GENERIC_TYPE,
    "type_arguments": NodeType.GENERIC_TYPE,
    "type_parameters": NodeType.OTHER,
    "reference_type": NodeType.TYPE_REF,
    "tuple_type": NodeType.TYPE_REF,
    "unit_type": NodeType.TYPE_REF,           # () type
    "abstract_type": NodeType.TYPE_REF,       # impl Trait
    "qualified_type": NodeType.TYPE_REF,      # <T as Trait>::Assoc
    "bracketed_type": NodeType.TYPE_REF,      # <T>
    "bounded_type": NodeType.TYPE_REF,        # T + Trait
    "associated_type": NodeType.TYPE_REF,
    "type_binding": NodeType.TYPE_REF,        # Item = T in where clause
    "function_type": NodeType.FUNC_TYPE,
    "where_clause": NodeType.OTHER,
    "where_predicate": NodeType.OTHER,        # individual where predicate
    "trait_bounds": NodeType.OTHER,           # : Trait1 + Trait2
    "lifetime": NodeType.OTHER,
    "tuple_struct_pattern": NodeType.OTHER,
    "tuple_pattern": NodeType.OTHER,
    "struct_pattern": NodeType.OTHER,
    "remaining_field_pattern": NodeType.OTHER, # .. in struct patterns
    "slice_pattern": NodeType.OTHER,          # [a, b, ..] pattern
    "field_pattern": NodeType.OTHER,          # Struct { field: pattern }
    "or_pattern": NodeType.OTHER,             # a | b pattern
    "mut_pattern": NodeType.OTHER,            # mut x in pattern

    # Visibility/modifiers
    "visibility_modifier": NodeType.MODIFIER,
    "function_modifiers": NodeType.MODIFIER,
    "mutable_specifier": NodeType.MODIFIER,   # mut keyword
    "inner_attribute_item": NodeType.ANNOTATION,
    "extern_crate_declaration": NodeType.IMPORT,

    # Control flow
    "else_clause": NodeType.BLOCK,            # else { .. }
    "match_block": NodeType.BLOCK,            # match { arms }
    "let_condition": NodeType.OTHER,          # if let Some(x) = ..

    # Other
    "use_declaration": NodeType.IMPORT,
    "use_as_clause": NodeType.IMPORT,
    "use_wildcard": NodeType.IMPORT,          # use foo::*
    "attribute_item": NodeType.ANNOTATION,
    "attribute": NodeType.ANNOTATION,         # inner attribute
    "doc_comment": NodeType.COMMENT,
    "outer_doc_comment_marker": NodeType.COMMENT,
    "inner_doc_comment_marker": NodeType.COMMENT,
    "line_comment": NodeType.COMMENT,
    "block_comment": NodeType.COMMENT,
    "token_tree": NodeType.OTHER,             # macro body tokens
}


def rust_node_to_type(node_type: str) -> NodeType:
    """Map a Rust tree-sitter node type to normalized NodeType."""
    return _RUST_MAP.get(node_type, NodeType.UNKNOWN)


# ── Kotlin AST node type mappings (from tree-sitter-kotlin) ──────────────────
# Faithful transliteration of kotlin_node_to_type from node_types.hpp

_KOTLIN_MAP: dict[str, NodeType] = {
    # Statements
    "source_file": NodeType.BLOCK,
    "statements": NodeType.BLOCK,
    "control_structure_body": NodeType.BLOCK,
    "if_expression": NodeType.IF,
    "for_statement": NodeType.FOR,
    "while_statement": NodeType.WHILE,
    "do_while_statement": NodeType.WHILE,
    "when_expression": NodeType.SWITCH,
    "when_entry": NodeType.BLOCK,             # case arm → block-like
    "when_condition": NodeType.OTHER,
    "jump_expression": NodeType.RETURN,       # return, throw, break, continue
    "try_expression": NodeType.TRY,
    "catch_block": NodeType.BLOCK,
    "finally_block": NodeType.BLOCK,

    # Assignments
    "assignment": NodeType.ASSIGN,
    "directly_assignable_expression": NodeType.ASSIGN,

    # Expressions
    "parenthesized_expression": NodeType.OTHER,
    "as_expression": NodeType.CAST,
    "is_expression": NodeType.OTHER,
    "check_expression": NodeType.OTHER,       # in, !in, is, !is
    "comparison_expression": NodeType.OTHER,
    "equality_expression": NodeType.OTHER,
    "conjunction_expression": NodeType.OTHER,  # &&
    "disjunction_expression": NodeType.OTHER,  # ||
    "additive_expression": NodeType.OTHER,
    "multiplicative_expression": NodeType.OTHER,
    "range_expression": NodeType.RANGE,
    "infix_expression": NodeType.OTHER,
    "elvis_expression": NodeType.OTHER,        # ?:
    "prefix_expression": NodeType.OTHER,
    "postfix_expression": NodeType.OTHER,
    "spread_expression": NodeType.OTHER,       # *array
    "string_literal": NodeType.STRING,
    "line_string_literal": NodeType.STRING,
    "multi_line_string_literal": NodeType.STRING,

    # Literals
    "simple_identifier": NodeType.VARIABLE,
    "identifier": NodeType.VARIABLE,           # raw identifier in some contexts
    "integer_literal": NodeType.NUMBER,
    "long_literal": NodeType.NUMBER,
    "hex_literal": NodeType.NUMBER,
    "bin_literal": NodeType.NUMBER,
    "unsigned_literal": NodeType.NUMBER,
    "real_literal": NodeType.NUMBER,
    "boolean_literal": NodeType.BOOLEAN,
    "character_literal": NodeType.CHAR,
    "null_literal": NodeType.NULL_LIT,

    # Function/Method calls
    "call_expression": NodeType.CALL,
    "navigation_expression": NodeType.FIELD_ACCESS,
    "indexing_expression": NodeType.INDEX,
    "call_suffix": NodeType.OTHER,
    "navigation_suffix": NodeType.OTHER,
    "indexing_suffix": NodeType.OTHER,

    # Declarations
    "function_declaration": NodeType.FUNCTION,
    "class_declaration": NodeType.CLASS,
    "object_declaration": NodeType.CLASS,
    "companion_object": NodeType.CLASS,
    "enum_class_body": NodeType.ENUM,
    "enum_entry": NodeType.VAR_DECL,
    "interface_declaration": NodeType.INTERFACE,
    "property_declaration": NodeType.VAR_DECL,
    "multi_variable_declaration": NodeType.VAR_DECL,  # val (a, b) = pair
    "variable_declaration": NodeType.VAR_DECL,
    "getter": NodeType.FUNCTION,
    "setter": NodeType.FUNCTION,
    "parameter": NodeType.PARAM,
    "function_value_parameters": NodeType.OTHER,
    "class_parameter": NodeType.PARAM,
    "type_parameter": NodeType.TYPE_PARAM,
    "type_parameters": NodeType.OTHER,
    "type_constraints": NodeType.OTHER,
    "primary_constructor": NodeType.FUNCTION,
    "secondary_constructor": NodeType.FUNCTION,
    "constructor_invocation": NodeType.CALL,
    "constructor_delegation_call": NodeType.CALL,
    "delegation_specifier": NodeType.TYPE_REF,
    "explicitly_typed_enum_entry": NodeType.VAR_DECL,

    # Lambda
    "lambda_literal": NodeType.LAMBDA,
    "anonymous_function": NodeType.LAMBDA,
    "annotated_lambda": NodeType.LAMBDA,       # lambda with annotations
    "lambda_parameters": NodeType.OTHER,

    # Types
    "type_identifier": NodeType.TYPE_REF,      # raw type name
    "user_type": NodeType.TYPE_REF,
    "simple_user_type": NodeType.TYPE_REF,
    "nullable_type": NodeType.NULLABLE_TYPE,
    "function_type": NodeType.FUNC_TYPE,
    "type_projection": NodeType.TYPE_REF,
    "type_alias": NodeType.VAR_DECL,

    # Other
    "import_header": NodeType.IMPORT,
    "import_list": NodeType.IMPORT,
    "package_header": NodeType.PACKAGE,
    "annotation": NodeType.ANNOTATION,
    "single_annotation": NodeType.ANNOTATION,
    "multi_annotation": NodeType.ANNOTATION,
    "file_annotation": NodeType.ANNOTATION,
    "modifier": NodeType.MODIFIER,
    "multiline_comment": NodeType.COMMENT,
    "line_comment": NodeType.COMMENT,
    "class_body": NodeType.BLOCK,
    "function_body": NodeType.BLOCK,
    "expression_body": NodeType.BLOCK,         # fun x() = expr
    "property_delegate": NodeType.OTHER,
    "type_arguments": NodeType.GENERIC_TYPE,
    "value_arguments": NodeType.OTHER,
    "value_argument": NodeType.OTHER,
    "this_expression": NodeType.VARIABLE,
    "super_expression": NodeType.VARIABLE,
    "object_literal": NodeType.OTHER,
    "modifiers": NodeType.MODIFIER,
    "visibility_modifier": NodeType.MODIFIER,
    "inheritance_modifier": NodeType.MODIFIER,
    "function_modifier": NodeType.MODIFIER,
    "platform_modifier": NodeType.MODIFIER,
    "member_modifier": NodeType.MODIFIER,
    "class_modifier": NodeType.MODIFIER,
    "parameter_modifier": NodeType.MODIFIER,
    "property_modifier": NodeType.MODIFIER,
    "label": NodeType.OTHER,
    "collection_literal": NodeType.OTHER,
    "interpolated_expression": NodeType.OTHER,  # string template ${...}
    "interpolated_identifier": NodeType.VARIABLE,  # string template $name
    "string_content": NodeType.STRING,          # inner text of strings
    "binding_pattern_kind": NodeType.OTHER,     # val/var keyword binding
    "when_subject": NodeType.OTHER,             # when(subject)
    "callable_reference": NodeType.FIELD_ACCESS,  # ::reference
    "type_test": NodeType.OTHER,                # is/!is check
    "type_parameter_modifiers": NodeType.MODIFIER,
    "reification_modifier": NodeType.MODIFIER,  # reified keyword
    "type_projection_modifiers": NodeType.MODIFIER,
    "variance_modifier": NodeType.MODIFIER,     # in/out
    "dynamic_type": NodeType.TYPE_REF,          # dynamic keyword type
    "function_type_parameters": NodeType.OTHER, # (T) -> R type params
}


def kotlin_node_to_type(node_type: str) -> NodeType:
    """Map a Kotlin tree-sitter node type to normalized NodeType."""
    return _KOTLIN_MAP.get(node_type, NodeType.UNKNOWN)


# ── C++ AST node type mappings (from tree-sitter-cpp) ────────────────────────
# Faithful transliteration of cpp_node_to_type from node_types.hpp

_CPP_MAP: dict[str, NodeType] = {
    # Statements
    "compound_statement": NodeType.BLOCK,
    "if_statement": NodeType.IF,
    "for_statement": NodeType.FOR,
    "for_range_loop": NodeType.FOR,
    "while_statement": NodeType.WHILE,
    "do_statement": NodeType.WHILE,
    "switch_statement": NodeType.SWITCH,
    "return_statement": NodeType.RETURN,
    "continue_statement": NodeType.CONTINUE,
    "break_statement": NodeType.BREAK,
    "try_statement": NodeType.TRY,
    "throw_statement": NodeType.THROW,
    "goto_statement": NodeType.GOTO,

    # Assignments
    "assignment_expression": NodeType.ASSIGN,
    "compound_assignment_expr": NodeType.ASSIGN,

    # Comparisons and binary ops
    "binary_expression": NodeType.UNKNOWN,
    "conditional_expression": NodeType.TERNARY,
    "unary_expression": NodeType.UNKNOWN,

    # Literals
    "identifier": NodeType.VARIABLE,
    "field_identifier": NodeType.VARIABLE,
    "namespace_identifier": NodeType.VARIABLE,
    "type_identifier": NodeType.TYPE_REF,
    "number_literal": NodeType.NUMBER,
    "string_literal": NodeType.STRING,
    "raw_string_literal": NodeType.STRING,
    "char_literal": NodeType.CHAR,
    "true": NodeType.BOOLEAN,
    "false": NodeType.BOOLEAN,
    "nullptr": NodeType.NULL_LIT,

    # Function/Method calls
    "call_expression": NodeType.CALL,
    "field_expression": NodeType.FIELD_ACCESS,
    "subscript_expression": NodeType.INDEX,

    # Declarations
    "function_definition": NodeType.FUNCTION,
    "function_declarator": NodeType.FUNCTION,
    "class_specifier": NodeType.CLASS,
    "struct_specifier": NodeType.STRUCT,
    "enum_specifier": NodeType.ENUM,
    "declaration": NodeType.VAR_DECL,
    "init_declarator": NodeType.VAR_DECL,
    "parameter_declaration": NodeType.PARAM,
    "template_parameter_list": NodeType.TYPE_PARAM,

    # Templates (generics)
    "template_declaration": NodeType.GENERIC_TYPE,
    "template_type": NodeType.GENERIC_TYPE,

    # Lambda
    "lambda_expression": NodeType.LAMBDA,

    # Types
    "primitive_type": NodeType.TYPE_REF,
    "qualified_identifier": NodeType.TYPE_REF,
    "pointer_declarator": NodeType.TYPE_REF,
    "reference_declarator": NodeType.TYPE_REF,
    "array_declarator": NodeType.ARRAY_TYPE,

    # Namespaces and includes
    "preproc_include": NodeType.IMPORT,
    "using_declaration": NodeType.IMPORT,
    "namespace_definition": NodeType.PACKAGE,
    "declaration_list": NodeType.PACKAGE,  # Map to PACKAGE to flatten structural nesting

    # Other
    "comment": NodeType.COMMENT,
    "attribute": NodeType.ANNOTATION,
    "storage_class_specifier": NodeType.MODIFIER,
    "type_qualifier": NodeType.MODIFIER,
    "virtual_specifier": NodeType.MODIFIER,
    "access_specifier": NodeType.MODIFIER,
    "linkage_specification": NodeType.PACKAGE,  # extern "C"
    "base_clause": NodeType.TYPE_REF,  # Inheritance list
    "parameter_list": NodeType.OTHER,
    "argument_list": NodeType.OTHER,
    "template_argument_list": NodeType.GENERIC_TYPE,
    "field_declaration": NodeType.VAR_DECL,
    "alias_declaration": NodeType.TYPE_REF,
}


def cpp_node_to_type(node_type: str) -> NodeType:
    """Map a C++ tree-sitter node type to normalized NodeType."""
    return _CPP_MAP.get(node_type, NodeType.UNKNOWN)


# ── Python AST node type mappings (from tree-sitter-python) ──────────────────
# Faithful transliteration of python_node_to_type from node_types.hpp

_PYTHON_MAP: dict[str, NodeType] = {
    # Statements / control flow
    "module": NodeType.BLOCK,
    "block": NodeType.BLOCK,
    "if_statement": NodeType.IF,
    "for_statement": NodeType.FOR,
    "while_statement": NodeType.WHILE,
    "match_statement": NodeType.SWITCH,
    "return_statement": NodeType.RETURN,
    "continue_statement": NodeType.CONTINUE,
    "break_statement": NodeType.BREAK,
    "try_statement": NodeType.TRY,
    "raise_statement": NodeType.THROW,

    # Assignments
    "assignment": NodeType.ASSIGN,
    "augmented_assignment": NodeType.ASSIGN,

    # Operators / expressions
    "binary_operator": NodeType.UNKNOWN,       # operator captured by unnamed children
    "unary_operator": NodeType.UNKNOWN,        # operator captured by unnamed children
    "boolean_operator": NodeType.UNKNOWN,      # operator captured by unnamed children
    "comparison_operator": NodeType.UNKNOWN,   # operator captured by unnamed children
    "call": NodeType.CALL,
    "attribute": NodeType.FIELD_ACCESS,
    "subscript": NodeType.INDEX,
    "lambda": NodeType.LAMBDA,
    "conditional_expression": NodeType.TERNARY,

    # Literals / identifiers
    "identifier": NodeType.VARIABLE,
    "integer": NodeType.NUMBER,
    "float": NodeType.NUMBER,
    "string": NodeType.STRING,
    "true": NodeType.BOOLEAN,
    "false": NodeType.BOOLEAN,
    "none": NodeType.NULL_LIT,

    # Declarations
    "function_definition": NodeType.FUNCTION,
    "class_definition": NodeType.CLASS,

    # Imports / comments
    "import_statement": NodeType.IMPORT,
    "import_from_statement": NodeType.IMPORT,
    "comment": NodeType.COMMENT,
}


def python_node_to_type(node_type: str) -> NodeType:
    """Map a Python tree-sitter node type to normalized NodeType."""
    return _PYTHON_MAP.get(node_type, NodeType.UNKNOWN)


# ── Utility ──────────────────────────────────────────────────────────────────
# Faithful transliteration of node_type_name from node_types.hpp

_NODE_TYPE_NAMES: dict[NodeType, str] = {
    NodeType.BLOCK: "BLOCK",
    NodeType.IF: "IF",
    NodeType.FOR: "FOR",
    NodeType.WHILE: "WHILE",
    NodeType.SWITCH: "SWITCH",
    NodeType.RETURN: "RETURN",
    NodeType.GOTO: "GOTO",
    NodeType.CONTINUE: "CONTINUE",
    NodeType.BREAK: "BREAK",
    NodeType.TRY: "TRY",
    NodeType.THROW: "THROW",
    NodeType.ASSIGN: "ASSIGN",
    NodeType.ASSIGN_ADD: "ASSIGN_ADD",
    NodeType.ASSIGN_SUB: "ASSIGN_SUB",
    NodeType.ASSIGN_MUL: "ASSIGN_MUL",
    NodeType.ASSIGN_DIV: "ASSIGN_DIV",
    NodeType.ASSIGN_MOD: "ASSIGN_MOD",
    NodeType.ASSIGN_AND: "ASSIGN_AND",
    NodeType.ASSIGN_OR: "ASSIGN_OR",
    NodeType.ASSIGN_XOR: "ASSIGN_XOR",
    NodeType.EQ: "EQ",
    NodeType.NE: "NE",
    NodeType.GT: "GT",
    NodeType.LT: "LT",
    NodeType.GE: "GE",
    NodeType.LE: "LE",
    NodeType.ADD: "ADD",
    NodeType.SUB: "SUB",
    NodeType.MUL: "MUL",
    NodeType.DIV: "DIV",
    NodeType.MOD: "MOD",
    NodeType.NEG: "NEG",
    NodeType.BIT_AND: "BIT_AND",
    NodeType.BIT_OR: "BIT_OR",
    NodeType.BIT_XOR: "BIT_XOR",
    NodeType.BIT_NOT: "BIT_NOT",
    NodeType.SHL: "SHL",
    NodeType.SHR: "SHR",
    NodeType.AND: "AND",
    NodeType.OR: "OR",
    NodeType.NOT: "NOT",
    NodeType.INDEX: "INDEX",
    NodeType.FIELD_ACCESS: "FIELD_ACCESS",
    NodeType.CALL: "CALL",
    NodeType.METHOD_CALL: "METHOD_CALL",
    NodeType.LAMBDA: "LAMBDA",
    NodeType.TERNARY: "TERNARY",
    NodeType.CAST: "CAST",
    NodeType.RANGE: "RANGE",
    NodeType.VARIABLE: "VARIABLE",
    NodeType.NUMBER: "NUMBER",
    NodeType.STRING: "STRING",
    NodeType.BOOLEAN: "BOOLEAN",
    NodeType.NULL_LIT: "NULL_LIT",
    NodeType.CHAR: "CHAR",
    NodeType.FUNCTION: "FUNCTION",
    NodeType.CLASS: "CLASS",
    NodeType.STRUCT: "STRUCT",
    NodeType.ENUM: "ENUM",
    NodeType.INTERFACE: "INTERFACE",
    NodeType.VAR_DECL: "VAR_DECL",
    NodeType.PARAM: "PARAM",
    NodeType.TYPE_PARAM: "TYPE_PARAM",
    NodeType.TYPE_REF: "TYPE_REF",
    NodeType.ARRAY_TYPE: "ARRAY_TYPE",
    NodeType.NULLABLE_TYPE: "NULLABLE_TYPE",
    NodeType.FUNC_TYPE: "FUNC_TYPE",
    NodeType.GENERIC_TYPE: "GENERIC_TYPE",
    NodeType.ARITHMETIC_OP: "ARITHMETIC_OP",
    NodeType.COMPARISON_OP: "COMPARISON_OP",
    NodeType.LOGICAL_OP: "LOGICAL_OP",
    NodeType.BITWISE_OP: "BITWISE_OP",
    NodeType.ASSIGNMENT_OP: "ASSIGNMENT_OP",
    NodeType.COMMENT: "COMMENT",
    NodeType.IMPORT: "IMPORT",
    NodeType.PACKAGE: "PACKAGE",
    NodeType.ANNOTATION: "ANNOTATION",
    NodeType.MODIFIER: "MODIFIER",
    NodeType.OTHER: "OTHER",
    NodeType.UNKNOWN: "UNKNOWN",
}


def node_type_name(type: NodeType) -> str:
    """Get human-readable name for a node type.

    Transliteration of node_type_name() from include/node_types.hpp:601-637.
    """
    return _NODE_TYPE_NAMES.get(type, "OTHER")
