// port-lint: source include/node_types.hpp
import { Language, NodeType } from "./types.js";

/**
 * Rust AST node type mappings (from tree-sitter-rust)
 */
export function rustNodeToType(nodeType: string): NodeType {
  const mapping: { [key: string]: NodeType } = {
    // Statements
    block: NodeType.BLOCK,
    if_expression: NodeType.IF,
    if_let_expression: NodeType.IF,
    for_expression: NodeType.FOR,
    while_expression: NodeType.WHILE,
    while_let_expression: NodeType.WHILE,
    loop_expression: NodeType.WHILE,
    match_expression: NodeType.SWITCH,
    match_arm: NodeType.BLOCK, // case arm → block-like
    match_pattern: NodeType.OTHER,
    return_expression: NodeType.RETURN,
    continue_expression: NodeType.CONTINUE,
    break_expression: NodeType.BREAK,
    try_expression: NodeType.TRY,
    expression_statement: NodeType.OTHER, // expression used as statement
    unsafe_block: NodeType.BLOCK,

    // Assignments
    assignment_expression: NodeType.ASSIGN,
    compound_assignment_expr: NodeType.ASSIGN,

    // Expressions
    binary_expression: NodeType.OTHER, // operator captured from unnamed children
    unary_expression: NodeType.OTHER,
    reference_expression: NodeType.OTHER, // &foo, &mut foo
    dereference_expression: NodeType.OTHER, // *foo
    range_expression: NodeType.RANGE,
    parenthesized_expression: NodeType.OTHER,
    array_expression: NodeType.OTHER, // array literal [a, b, c]
    tuple_expression: NodeType.OTHER,
    unit_expression: NodeType.OTHER, // ()
    type_cast_expression: NodeType.CAST,

    // Literals
    identifier: NodeType.VARIABLE,
    field_identifier: NodeType.VARIABLE, // field name in field_expression
    integer_literal: NodeType.NUMBER,
    float_literal: NodeType.NUMBER,
    string_literal: NodeType.STRING,
    raw_string_literal: NodeType.STRING,
    string_content: NodeType.STRING, // inner text of string literals
    boolean_literal: NodeType.BOOLEAN,
    char_literal: NodeType.CHAR,
    self: NodeType.VARIABLE, // self keyword → variable
    super: NodeType.OTHER, // super keyword in paths
    crate: NodeType.OTHER, // crate keyword in paths
    shorthand_field_identifier: NodeType.VARIABLE, // shorthand field in patterns

    // Function/Method calls
    call_expression: NodeType.CALL,
    method_call_expression: NodeType.METHOD_CALL,
    macro_invocation: NodeType.CALL, // macro!() → call-like
    generic_function: NodeType.CALL, // foo::<T>() → call-like
    arguments: NodeType.OTHER,
    parameters: NodeType.OTHER,
    closure_parameters: NodeType.OTHER, // |x, y| closure params

    // Declarations
    source_file: NodeType.BLOCK, // root module → block
    function_item: NodeType.FUNCTION,
    struct_item: NodeType.STRUCT,
    enum_item: NodeType.ENUM,
    enum_variant_list: NodeType.BLOCK,
    enum_variant: NodeType.VAR_DECL,
    trait_item: NodeType.INTERFACE,
    impl_item: NodeType.BLOCK, // Treat like block (matches Kotlin class_body → BLOCK)
    declaration_list: NodeType.BLOCK, // impl body — maps to Kotlin class_body (also BLOCK)
    let_declaration: NodeType.VAR_DECL,
    const_item: NodeType.VAR_DECL,
    static_item: NodeType.VAR_DECL,
    type_item: NodeType.VAR_DECL, // type alias
    function_signature_item: NodeType.FUNCTION, // trait method signature
    parameter: NodeType.PARAM,
    self_parameter: NodeType.PARAM,
    type_parameter: NodeType.TYPE_PARAM,
    constrained_type_parameter: NodeType.TYPE_PARAM,
    field_declaration_list: NodeType.BLOCK,
    ordered_field_declaration_list: NodeType.BLOCK, // tuple struct fields
    field_declaration: NodeType.VAR_DECL,
    struct_expression: NodeType.CALL, // Struct { .. } → constructor call-like
    field_initializer_list: NodeType.OTHER,
    field_initializer: NodeType.ASSIGN,
    shorthand_field_initializer: NodeType.ASSIGN, // Struct { field } shorthand
    base_field_initializer: NodeType.ASSIGN, // ..base
    mod_item: NodeType.PACKAGE,

    // Field/Index access
    field_expression: NodeType.FIELD_ACCESS,
    index_expression: NodeType.INDEX,

    // Closure
    closure_expression: NodeType.LAMBDA,

    // Types
    type_identifier: NodeType.TYPE_REF,
    primitive_type: NodeType.TYPE_REF, // i32, bool, str etc
    scoped_identifier: NodeType.FIELD_ACCESS, // path::to::Type → field access chain
    scoped_type_identifier: NodeType.TYPE_REF,
    array_type: NodeType.ARRAY_TYPE,
    generic_type: NodeType.GENERIC_TYPE,
    generic_type_with_turbofish: NodeType.GENERIC_TYPE,
    type_arguments: NodeType.GENERIC_TYPE,
    type_parameters: NodeType.OTHER,
    reference_type: NodeType.TYPE_REF,
    tuple_type: NodeType.TYPE_REF,
    unit_type: NodeType.TYPE_REF, // () type
    abstract_type: NodeType.TYPE_REF, // impl Trait
    qualified_type: NodeType.TYPE_REF, // <T as Trait>::Assoc
    bracketed_type: NodeType.TYPE_REF, // <T>
    bounded_type: NodeType.TYPE_REF, // T + Trait
    associated_type: NodeType.TYPE_REF,
    type_binding: NodeType.TYPE_REF, // Item = T in where clause
    function_type: NodeType.FUNC_TYPE,
    where_clause: NodeType.OTHER,
    where_predicate: NodeType.OTHER, // individual where predicate
    trait_bounds: NodeType.OTHER, // : Trait1 + Trait2
    lifetime: NodeType.OTHER,
    tuple_struct_pattern: NodeType.OTHER,
    tuple_pattern: NodeType.OTHER,
    struct_pattern: NodeType.OTHER,
    remaining_field_pattern: NodeType.OTHER, // .. in struct patterns
    slice_pattern: NodeType.OTHER, // [a, b, ..] pattern
    field_pattern: NodeType.OTHER, // Struct { field: pattern }
    or_pattern: NodeType.OTHER, // a | b pattern
    mut_pattern: NodeType.OTHER, // mut x in pattern

    // Visibility/modifiers
    visibility_modifier: NodeType.MODIFIER,
    function_modifiers: NodeType.MODIFIER,
    mutable_specifier: NodeType.MODIFIER, // mut keyword
    inner_attribute_item: NodeType.ANNOTATION,
    extern_crate_declaration: NodeType.IMPORT,

    // Control flow
    else_clause: NodeType.BLOCK, // else { .. }
    match_block: NodeType.BLOCK, // match { arms }
    let_condition: NodeType.OTHER, // if let Some(x) = ..

    // Other
    use_declaration: NodeType.IMPORT,
    use_as_clause: NodeType.IMPORT,
    use_wildcard: NodeType.IMPORT, // use foo::*
    attribute_item: NodeType.ANNOTATION,
    attribute: NodeType.ANNOTATION, // inner attribute
    doc_comment: NodeType.COMMENT,
    outer_doc_comment_marker: NodeType.COMMENT,
    inner_doc_comment_marker: NodeType.COMMENT,
    line_comment: NodeType.COMMENT,
    block_comment: NodeType.COMMENT,
    token_tree: NodeType.OTHER, // macro body tokens
    // Tree-sitter-rust internal placeholder nodes which are not semantically meaningful.
    removed_trait_bound: NodeType.OTHER,
  };

  return mapping[nodeType] ?? NodeType.UNKNOWN;
}

/**
 * Kotlin AST node type mappings (from tree-sitter-kotlin)
 */
export function kotlinNodeToType(nodeType: string): NodeType {
  const mapping: { [key: string]: NodeType } = {
    // Statements
    source_file: NodeType.BLOCK,
    statements: NodeType.BLOCK,
    control_structure_body: NodeType.BLOCK,
    if_expression: NodeType.IF,
    for_statement: NodeType.FOR,
    while_statement: NodeType.WHILE,
    do_while_statement: NodeType.WHILE,
    when_expression: NodeType.SWITCH,
    when_entry: NodeType.BLOCK, // case arm → block-like
    when_condition: NodeType.OTHER,
    jump_expression: NodeType.RETURN, // return, throw, break, continue
    try_expression: NodeType.TRY,
    catch_block: NodeType.BLOCK,
    finally_block: NodeType.BLOCK,

    // Assignments
    assignment: NodeType.ASSIGN,
    directly_assignable_expression: NodeType.ASSIGN,

    // Expressions
    parenthesized_expression: NodeType.OTHER,
    parenthesized_type: NodeType.TYPE_REF,
    as_expression: NodeType.CAST,
    is_expression: NodeType.OTHER,
    check_expression: NodeType.OTHER, // in, !in, is, !is
    comparison_expression: NodeType.OTHER,
    equality_expression: NodeType.OTHER,
    conjunction_expression: NodeType.OTHER, // &&
    disjunction_expression: NodeType.OTHER, // ||
    additive_expression: NodeType.OTHER,
    multiplicative_expression: NodeType.OTHER,
    range_expression: NodeType.RANGE,
    infix_expression: NodeType.OTHER,
    elvis_expression: NodeType.OTHER, // ?:
    prefix_expression: NodeType.OTHER,
    postfix_expression: NodeType.OTHER,
    spread_expression: NodeType.OTHER, // *array
    string_literal: NodeType.STRING,
    line_string_literal: NodeType.STRING,
    multi_line_string_literal: NodeType.STRING,

    // Literals
    simple_identifier: NodeType.VARIABLE,
    identifier: NodeType.VARIABLE, // raw identifier in some contexts
    integer_literal: NodeType.NUMBER,
    long_literal: NodeType.NUMBER,
    hex_literal: NodeType.NUMBER,
    bin_literal: NodeType.NUMBER,
    unsigned_literal: NodeType.NUMBER,
    real_literal: NodeType.NUMBER,
    boolean_literal: NodeType.BOOLEAN,
    character_literal: NodeType.CHAR,
    null_literal: NodeType.NULL_LIT,

    // Function/Method calls
    call_expression: NodeType.CALL,
    navigation_expression: NodeType.FIELD_ACCESS,
    indexing_expression: NodeType.INDEX,
    call_suffix: NodeType.OTHER,
    navigation_suffix: NodeType.OTHER,
    indexing_suffix: NodeType.OTHER,

    // Declarations
    function_declaration: NodeType.FUNCTION,
    class_declaration: NodeType.CLASS,
    object_declaration: NodeType.CLASS,
    companion_object: NodeType.CLASS,
    enum_class_body: NodeType.ENUM,
    enum_entry: NodeType.VAR_DECL,
    interface_declaration: NodeType.INTERFACE,
    property_declaration: NodeType.VAR_DECL,
    multi_variable_declaration: NodeType.VAR_DECL, // val (a, b) = pair
    variable_declaration: NodeType.VAR_DECL,
    getter: NodeType.FUNCTION,
    setter: NodeType.FUNCTION,
    parameter: NodeType.PARAM,
    function_value_parameters: NodeType.OTHER,
    class_parameter: NodeType.PARAM,
    type_parameter: NodeType.TYPE_PARAM,
    type_parameters: NodeType.OTHER,
    type_constraints: NodeType.OTHER,
    primary_constructor: NodeType.FUNCTION,
    secondary_constructor: NodeType.FUNCTION,
    constructor_invocation: NodeType.CALL,
    constructor_delegation_call: NodeType.CALL,
    delegation_specifier: NodeType.TYPE_REF,
    explicitly_typed_enum_entry: NodeType.VAR_DECL,

    // Lambda
    lambda_literal: NodeType.LAMBDA,
    anonymous_function: NodeType.LAMBDA,
    annotated_lambda: NodeType.LAMBDA, // lambda with annotations
    lambda_parameters: NodeType.OTHER,

    // Types
    type_identifier: NodeType.TYPE_REF, // raw type name
    user_type: NodeType.TYPE_REF,
    simple_user_type: NodeType.TYPE_REF,
    nullable_type: NodeType.NULLABLE_TYPE,
    function_type: NodeType.FUNC_TYPE,
    type_projection: NodeType.TYPE_REF,
    type_alias: NodeType.VAR_DECL,

    // Other
    import_header: NodeType.IMPORT,
    import_list: NodeType.IMPORT,
    import_alias: NodeType.IMPORT,
    package_header: NodeType.PACKAGE,
    annotation: NodeType.ANNOTATION,
    single_annotation: NodeType.ANNOTATION,
    multi_annotation: NodeType.ANNOTATION,
    file_annotation: NodeType.ANNOTATION,
    modifier: NodeType.MODIFIER,
    multiline_comment: NodeType.COMMENT,
    line_comment: NodeType.COMMENT,
    class_body: NodeType.BLOCK,
    function_body: NodeType.BLOCK,
    expression_body: NodeType.BLOCK, // fun x() = expr
    property_delegate: NodeType.OTHER,
    type_arguments: NodeType.GENERIC_TYPE,
    value_arguments: NodeType.OTHER,
    value_argument: NodeType.OTHER,
    this_expression: NodeType.VARIABLE,
    super_expression: NodeType.VARIABLE,
    object_literal: NodeType.OTHER,
    modifiers: NodeType.MODIFIER,
    visibility_modifier: NodeType.MODIFIER,
    inheritance_modifier: NodeType.MODIFIER,
    function_modifier: NodeType.MODIFIER,
    platform_modifier: NodeType.MODIFIER,
    member_modifier: NodeType.MODIFIER,
    class_modifier: NodeType.MODIFIER,
    parameter_modifier: NodeType.MODIFIER,
    property_modifier: NodeType.MODIFIER,
    label: NodeType.OTHER,
    collection_literal: NodeType.OTHER,
    interpolated_expression: NodeType.OTHER, // string template ${...}
    interpolated_identifier: NodeType.VARIABLE, // string template $name
    string_content: NodeType.STRING, // inner text of strings
    binding_pattern_kind: NodeType.OTHER, // val/var keyword binding
    when_subject: NodeType.OTHER, // when(subject)
    callable_reference: NodeType.FIELD_ACCESS, // ::reference
    type_test: NodeType.OTHER, // is/!is check
    type_parameter_modifiers: NodeType.MODIFIER,
    reification_modifier: NodeType.MODIFIER, // reified keyword
    type_projection_modifiers: NodeType.MODIFIER,
    variance_modifier: NodeType.MODIFIER, // in/out
    dynamic_type: NodeType.TYPE_REF, // dynamic keyword type
    function_type_parameters: NodeType.OTHER, // (T) -> R type params
  };

  return mapping[nodeType] ?? NodeType.UNKNOWN;
}

/**
 * C++ AST node type mappings (from tree-sitter-cpp)
 */
export function cppNodeToType(nodeType: string): NodeType {
  const mapping: { [key: string]: NodeType } = {
    // Statements
    compound_statement: NodeType.BLOCK,
    if_statement: NodeType.IF,
    for_statement: NodeType.FOR,
    for_range_loop: NodeType.FOR,
    while_statement: NodeType.WHILE,
    do_statement: NodeType.WHILE,
    switch_statement: NodeType.SWITCH,
    return_statement: NodeType.RETURN,
    continue_statement: NodeType.CONTINUE,
    break_statement: NodeType.BREAK,
    try_statement: NodeType.TRY,
    throw_statement: NodeType.THROW,
    goto_statement: NodeType.GOTO,

    // Assignments
    assignment_expression: NodeType.ASSIGN,
    compound_assignment_expr: NodeType.ASSIGN,

    // Comparisons and binary ops
    binary_expression: NodeType.UNKNOWN,
    conditional_expression: NodeType.TERNARY,
    unary_expression: NodeType.UNKNOWN,

    // Literals
    identifier: NodeType.VARIABLE,
    field_identifier: NodeType.VARIABLE,
    namespace_identifier: NodeType.VARIABLE,
    type_identifier: NodeType.TYPE_REF,
    number_literal: NodeType.NUMBER,
    string_literal: NodeType.STRING,
    raw_string_literal: NodeType.STRING,
    char_literal: NodeType.CHAR,
    true: NodeType.BOOLEAN,
    false: NodeType.BOOLEAN,
    nullptr: NodeType.NULL_LIT,

    // Function/Method calls
    call_expression: NodeType.CALL,
    field_expression: NodeType.FIELD_ACCESS,
    subscript_expression: NodeType.INDEX,

    // Declarations
    function_definition: NodeType.FUNCTION,
    function_declarator: NodeType.FUNCTION,
    class_specifier: NodeType.CLASS,
    struct_specifier: NodeType.STRUCT,
    enum_specifier: NodeType.ENUM,
    declaration: NodeType.VAR_DECL,
    init_declarator: NodeType.VAR_DECL,
    parameter_declaration: NodeType.PARAM,
    template_parameter_list: NodeType.TYPE_PARAM,

    // Templates (generics)
    template_declaration: NodeType.GENERIC_TYPE,
    template_type: NodeType.GENERIC_TYPE,

    // Lambda
    lambda_expression: NodeType.LAMBDA,

    // Types
    primitive_type: NodeType.TYPE_REF,
    qualified_identifier: NodeType.TYPE_REF,
    pointer_declarator: NodeType.TYPE_REF,
    reference_declarator: NodeType.TYPE_REF,
    array_declarator: NodeType.ARRAY_TYPE,

    // Namespaces and includes
    preproc_include: NodeType.IMPORT,
    using_declaration: NodeType.IMPORT,
    namespace_definition: NodeType.PACKAGE,
    declaration_list: NodeType.PACKAGE, // Map to PACKAGE to flatten structural nesting

    // Other
    comment: NodeType.COMMENT,
    attribute: NodeType.ANNOTATION,
    storage_class_specifier: NodeType.MODIFIER,
    type_qualifier: NodeType.MODIFIER,
    virtual_specifier: NodeType.MODIFIER,
    access_specifier: NodeType.MODIFIER,
    linkage_specification: NodeType.PACKAGE, // extern "C"
    base_clause: NodeType.TYPE_REF, // Inheritance list
    parameter_list: NodeType.OTHER,
    argument_list: NodeType.OTHER,
    template_argument_list: NodeType.GENERIC_TYPE,
    field_declaration: NodeType.VAR_DECL,
    alias_declaration: NodeType.TYPE_REF,
  };

  return mapping[nodeType] ?? NodeType.UNKNOWN;
}

/**
 * Python AST node type mappings (from tree-sitter-python)
 */
export function pythonNodeToType(nodeType: string): NodeType {
  const mapping: { [key: string]: NodeType } = {
    // Statements / control flow
    module: NodeType.BLOCK,
    block: NodeType.BLOCK,
    if_statement: NodeType.IF,
    for_statement: NodeType.FOR,
    while_statement: NodeType.WHILE,
    match_statement: NodeType.SWITCH,
    return_statement: NodeType.RETURN,
    continue_statement: NodeType.CONTINUE,
    break_statement: NodeType.BREAK,
    try_statement: NodeType.TRY,
    raise_statement: NodeType.THROW,

    // Assignments
    assignment: NodeType.ASSIGN,
    augmented_assignment: NodeType.ASSIGN,

    // Operators / expressions
    binary_operator: NodeType.UNKNOWN, // operator captured by unnamed children
    unary_operator: NodeType.UNKNOWN, // operator captured by unnamed children
    boolean_operator: NodeType.UNKNOWN, // operator captured by unnamed children
    comparison_operator: NodeType.UNKNOWN, // operator captured by unnamed children
    call: NodeType.CALL,
    attribute: NodeType.FIELD_ACCESS,
    subscript: NodeType.INDEX,
    lambda: NodeType.LAMBDA,
    conditional_expression: NodeType.TERNARY,

    // Literals / identifiers
    identifier: NodeType.VARIABLE,
    integer: NodeType.NUMBER,
    float: NodeType.NUMBER,
    string: NodeType.STRING,
    true: NodeType.BOOLEAN,
    false: NodeType.BOOLEAN,
    none: NodeType.NULL_LIT,

    // Declarations
    function_definition: NodeType.FUNCTION,
    class_definition: NodeType.CLASS,

    // Imports / comments
    import_statement: NodeType.IMPORT,
    import_from_statement: NodeType.IMPORT,
    comment: NodeType.COMMENT,
  };

  return mapping[nodeType] ?? NodeType.UNKNOWN;
}

/**
 * TypeScript AST node type mappings (from tree-sitter-typescript)
 */
export function typescriptNodeToType(nodeType: string): NodeType {
  const mapping: { [key: string]: NodeType } = {
    // Statements
    source_file: NodeType.BLOCK,
    program: NodeType.BLOCK,
    block: NodeType.BLOCK,
    statement_block: NodeType.BLOCK,
    if_statement: NodeType.IF,
    for_statement: NodeType.FOR,
    for_in_statement: NodeType.FOR,
    for_of_statement: NodeType.FOR,
    while_statement: NodeType.WHILE,
    do_statement: NodeType.WHILE,
    switch_statement: NodeType.SWITCH,
    switch_case: NodeType.SWITCH,
    case: NodeType.SWITCH,
    break_statement: NodeType.BREAK,
    continue_statement: NodeType.CONTINUE,
    return_statement: NodeType.RETURN,
    throw_statement: NodeType.THROW,
    try_statement: NodeType.TRY,
    catch_clause: NodeType.TRY,
    finally_clause: NodeType.BLOCK,

    // Assignments
    assignment_expression: NodeType.ASSIGN,
    augmented_assignment_expression: NodeType.ASSIGN,

    // Expressions
    binary_expression: NodeType.UNKNOWN,
    unary_expression: NodeType.OTHER,
    update_expression: NodeType.OTHER,
    ternary_expression: NodeType.TERNARY,
    parenthesized_expression: NodeType.OTHER,
    as_expression: NodeType.CAST,

    // Literals
    identifier: NodeType.VARIABLE,
    property_identifier: NodeType.VARIABLE,
    type_identifier: NodeType.TYPE_REF,
    number: NodeType.NUMBER,
    string: NodeType.STRING,
    template_string: NodeType.STRING,
    regex: NodeType.STRING,
    true: NodeType.BOOLEAN,
    false: NodeType.BOOLEAN,
    null: NodeType.NULL_LIT,
    undefined: NodeType.NULL_LIT,
    this: NodeType.VARIABLE,
    super: NodeType.VARIABLE,

    // Function/Method calls
    call_expression: NodeType.CALL,
    new_expression: NodeType.CALL,
    member_expression: NodeType.FIELD_ACCESS,
    subscript_expression: NodeType.INDEX,

    // Declarations
    function_declaration: NodeType.FUNCTION,
    function_expression: NodeType.FUNCTION,
    arrow_function: NodeType.FUNCTION,
    method_definition: NodeType.FUNCTION,
    class_declaration: NodeType.CLASS,
    class_expression: NodeType.CLASS,
    interface_declaration: NodeType.INTERFACE,
    type_alias_declaration: NodeType.VAR_DECL,
    enum_declaration: NodeType.ENUM,
    lexical_declaration: NodeType.VAR_DECL,
    variable_declaration: NodeType.VAR_DECL,
    formal_parameters: NodeType.OTHER,
    required_parameter: NodeType.PARAM,
    optional_parameter: NodeType.PARAM,
    rest_parameter: NodeType.PARAM,
    type_parameter: NodeType.TYPE_PARAM,

    // Types
    type_annotation: NodeType.TYPE_REF,
    predefined_type: NodeType.TYPE_REF,
    union_type: NodeType.TYPE_REF,
    intersection_type: NodeType.TYPE_REF,
    array_type: NodeType.ARRAY_TYPE,
    generic_type: NodeType.GENERIC_TYPE,
    type_arguments: NodeType.GENERIC_TYPE,

    // Import/Export
    import_statement: NodeType.IMPORT,
    export_statement: NodeType.OTHER,
    import_clause: NodeType.IMPORT,
    export_clause: NodeType.OTHER,

    // Comments
    comment: NodeType.COMMENT,
    line_comment: NodeType.COMMENT,
    block_comment: NodeType.COMMENT,

    // Other
    property_signature: NodeType.VAR_DECL,
    property_definition: NodeType.VAR_DECL,
    class_body: NodeType.BLOCK,
    decorator: NodeType.ANNOTATION,
  };

  return mapping[nodeType] ?? NodeType.UNKNOWN;
}

/**
 * Get human-readable name for a node type
 */
export function nodeTypeName(type: NodeType): string {
  switch (type) {
    case NodeType.BLOCK:
      return "BLOCK";
    case NodeType.IF:
      return "IF";
    case NodeType.FOR:
      return "FOR";
    case NodeType.WHILE:
      return "WHILE";
    case NodeType.SWITCH:
      return "SWITCH";
    case NodeType.RETURN:
      return "RETURN";
    case NodeType.CONTINUE:
      return "CONTINUE";
    case NodeType.BREAK:
      return "BREAK";
    case NodeType.TRY:
      return "TRY";
    case NodeType.THROW:
      return "THROW";
    case NodeType.ASSIGN:
      return "ASSIGN";
    case NodeType.EQ:
      return "EQ";
    case NodeType.NE:
      return "NE";
    case NodeType.GT:
      return "GT";
    case NodeType.LT:
      return "LT";
    case NodeType.ADD:
      return "ADD";
    case NodeType.SUB:
      return "SUB";
    case NodeType.MUL:
      return "MUL";
    case NodeType.DIV:
      return "DIV";
    case NodeType.CALL:
      return "CALL";
    case NodeType.METHOD_CALL:
      return "METHOD_CALL";
    case NodeType.VARIABLE:
      return "VARIABLE";
    case NodeType.NUMBER:
      return "NUMBER";
    case NodeType.STRING:
      return "STRING";
    case NodeType.FUNCTION:
      return "FUNCTION";
    case NodeType.CLASS:
      return "CLASS";
    case NodeType.STRUCT:
      return "STRUCT";
    case NodeType.VAR_DECL:
      return "VAR_DECL";
    case NodeType.LAMBDA:
      return "LAMBDA";
    case NodeType.FIELD_ACCESS:
      return "FIELD_ACCESS";
    case NodeType.INDEX:
      return "INDEX";
    case NodeType.UNKNOWN:
      return "UNKNOWN";
    default:
      return "OTHER";
  }
}

/**
 * Get node type map for language (Unified function for ASTParser parity)
 */
export function normalizeNodeType(nodeType: string, language: Language): NodeType {
  switch (language) {
    case Language.TYPESCRIPT:
      return typescriptNodeToType(nodeType);
    case Language.RUST:
      return rustNodeToType(nodeType);
    case Language.KOTLIN:
      return kotlinNodeToType(nodeType);
    case Language.CPP:
    case Language.C:
      return cppNodeToType(nodeType);
    case Language.PYTHON:
      return pythonNodeToType(nodeType);
    default:
      return NodeType.UNKNOWN;
  }
}
