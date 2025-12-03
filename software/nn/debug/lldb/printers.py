#
# Pretty-printers for Eigen and C++ STL types for LLDB
#
# This file is based on the GDB pretty-printers for Eigen and adapted for LLDB.
#
# To use, add the following to your ~/.lldbinit file:
# command script import /path/to/this/file/printers.py
#

import lldb

def __lldb_init_module(debugger, internal_dict):
    # Eigen formatters
    debugger.HandleCommand('type summary add -F printers.EigenMatrix_SummaryProvider "Eigen::Matrix" -r')
    debugger.HandleCommand('type summary add -F printers.EigenMatrix_SummaryProvider "Eigen::Array" -r')
    debugger.HandleCommand('type summary add -F printers.EigenQuaternion_SummaryProvider "Eigen::Quaternion" -r')
    debugger.HandleCommand('type synthetic add -l printers.EigenMatrix_SyntheticChildrenProvider "Eigen::Matrix" -r')
    debugger.HandleCommand('type synthetic add -l printers.EigenMatrix_SyntheticChildrenProvider "Eigen::Array" -r')

    # STL formatters
    # std::optional (for libc++ and libstdc++)
    debugger.HandleCommand('type summary add -x "std::optional<" --summary-string "Some(${var.__val_})" -w stl --name libcxx_optional_summary -v --child __has_val_ -C yes')
    debugger.HandleCommand('type summary add -x "std::optional<" --summary-string "None" -w stl --name libcxx_optional_summary_empty -v --child __has_val_ -C no')
    debugger.HandleCommand('type summary add -x "std::optional<" --summary-string "Some(${var._M_payload._M_payload._M_value})" -w stl --name libstdcxx_optional_summary -v --child _M_payload._M_engaged -C yes')
    debugger.HandleCommand('type summary add -x "std::optional<" --summary-string "None" -w stl --name libstdcxx_optional_summary_empty -v --child _M_payload._M_engaged -C no')

    # std::variant
    debugger.HandleCommand('type summary add -F printers.StdVariant_SummaryProvider -e -w stl -x "std::variant<"')


def StdVariant_SummaryProvider(valobj, internal_dict):
    """Provide a summary for std::variant for both libstdc++ and libc++."""
    # Common case: valueless by exception
    index_val = valobj.GetChildMemberWithName('_M_index') # libstdc++
    if not index_val.IsValid():
        index_val = valobj.GetChildMemberWithName('__index_') # libc++
    
    if not index_val.IsValid():
        return "std::variant (unknown implementation)"

    index = index_val.GetValueAsSigned(-2) # Use -2 as a sentinel for error
    if index == -1:
        return "valueless_by_exception"

    variant_type = valobj.GetType().GetUnqualifiedType()
    if index >= variant_type.GetNumberOfTemplateArguments():
        return "invalid_variant (index out of bounds)"

    active_type = variant_type.GetTemplateArgumentType(index)
    
    try:
        # libstdc++: _M_storage._M_first, _M_storage._M_rest...
        if valobj.GetChildMemberWithName('_M_storage'):
            storage = valobj.GetChildMemberWithName('_M_storage')
            if index == 0:
                data = storage.GetChildMemberWithName('_M_first')
            else:
                data = storage.GetChildMemberWithName('_M_rest')
                for _ in range(index - 1):
                    data = data.GetChildMemberWithName('_M_rest')
                data = data.GetChildMemberWithName('_M_first')
            
            value = data.Cast(active_type)
            return f'holds {active_type.GetName()} = {value.GetSummary() or value.GetValue()}'

        # libc++: __u_ union
        elif valobj.GetChildMemberWithName('__u_'):
            union = valobj.GetChildMemberWithName('__u_')
            # The active member is usually the first one of the correct type.
            value = union.GetChildAtIndex(0).Cast(active_type)
            return f'holds {active_type.GetName()} = {value.GetSummary() or value.GetValue()}'

    except Exception as e:
        return f'holds {active_type.GetName()} (error: {e})'

    return f'std::variant with index {index}'

def EigenMatrix_SummaryProvider(valobj, internal_dict):
    """Provide a summary for Eigen matrices and arrays"""
    try:
        rows, cols, is_row_major, type_name = get_matrix_info(valobj)
        variety = "Matrix"
        if "Array" in valobj.GetType().GetName():
            variety = "Array"
        
        storage_order = "RowMajor" if is_row_major else "ColMajor"
        return "Eigen::%s<%s, %d, %d, %s>" % (variety, type_name, rows, cols, storage_order)
    except Exception as e:
        return "Eigen::Matrix (error: %s)" % str(e)


def EigenQuaternion_SummaryProvider(valobj, internal_dict):
    """Provide a summary for Eigen quaternions"""
    try:
        coeffs = valobj.GetChildMemberWithName('m_coeffs')
        x = coeffs.GetChildAtIndex(0).GetValueAsSigned(0)
        y = coeffs.GetChildAtIndex(1).GetValueAsSigned(0)
        z = coeffs.GetChildAtIndex(2).GetValueAsSigned(0)
        w = coeffs.GetChildAtIndex(3).GetValueAsSigned(0)
        return "(w=%f, x=%f, y=%f, z=%f)" % (w, x, y, z)
    except Exception as e:
        return "Eigen::Quaternion (error: %s)" % str(e)

def get_matrix_info(valobj):
    """Extracts matrix properties from an Eigen::Matrix object."""
    # The type is buried in template arguments.
    # e.g. Eigen::Matrix<double, 4, 4, 0, 4, 4>
    type_name = valobj.GetType().GetTemplateArgumentType(0).GetName()
    
    # Get storage from the m_storage member
    storage = valobj.GetChildMemberWithName('m_storage')
    
    # Get rows and cols
    rows = storage.GetChildMemberWithName('m_rows').GetValueAsSigned(0)
    cols = storage.GetChildMemberWithName('m_cols').GetValueAsSigned(0)
    
    # If rows or cols are -1, it's a dynamic matrix.
    # The actual dimensions are stored in m_storage for dynamic matrices.
    # For fixed-size matrices, they are part of the type.
    if rows == -1 or rows == 4294967295: # -1 signed is (2^32 - 1) unsigned
        rows = storage.GetChildMemberWithName('m_rows').GetValueAsSigned(0)
    if cols == -1 or cols == 4294967295:
        cols = storage.GetChildMemberWithName('m_cols').GetValueAsSigned(0)

    # Determine storage order (RowMajor bit flag)
    # The options are in the 3rd template argument
    options = valobj.GetType().GetTemplateArgumentAsSigned(3, -1)
    is_row_major = (options & 1) != 0

    return int(rows), int(cols), is_row_major, type_name


class EigenMatrix_SyntheticChildrenProvider:
    """Provide synthetic children for Eigen matrices."""

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.rows = 0
        self.cols = 0
        self.is_row_major = False
        self.type_name = ""
        self.data_ptr = None

    def update(self):
        try:
            self.rows, self.cols, self.is_row_major, self.type_name = get_matrix_info(self.valobj)
            
            # Find the data pointer
            storage = self.valobj.GetChildMemberWithName('m_storage')
            m_data = storage.GetChildMemberWithName('m_data')

            # For fixed-size matrices, data is in an 'array' inside m_data struct
            if m_data.GetType().IsReference() or m_data.GetType().IsPointer():
                 self.data_ptr = m_data.AddressOf()
            elif 'array' in (c.GetName() for c in m_data.GetChildren()):
                self.data_ptr = m_data.GetChildMemberWithName('array').GetData()
            else: # Dynamic matrices
                self.data_ptr = m_data.GetData()

            if self.data_ptr:
                 self.element_type = self.valobj.GetType().GetTemplateArgumentType(0)
                 self.element_size = self.element_type.GetByteSize()
                 # Make sure we have a valid pointer
                 self.data_address = self.data_ptr.GetLoadAddress()
            else:
                self.data_address = lldb.LLDB_INVALID_ADDRESS

        except Exception:
            # If something goes wrong, we'll have no children
            self.rows = 0
            self.cols = 0
            self.data_address = lldb.LLDB_INVALID_ADDRESS

    def num_children(self):
        return self.rows * self.cols

    def get_child_index(self, name):
        try:
            # Children are named "[row,col]"
            if name.startswith('[') and name.endswith(']'):
                parts = name[1:-1].split(',')
                if len(parts) == 2:
                    row = int(parts[0])
                    col = int(parts[1])
                    if self.is_row_major:
                        return row * self.cols + col
                    else:
                        return col * self.rows + row
            # Or just index for vectors
            elif name.startswith('[') and name.endswith(']'):
                return int(name[1:-1])

        except:
            pass
        return -1

    def get_child_at_index(self, index):
        if self.data_address == lldb.LLDB_INVALID_ADDRESS:
            return None
        
        if index >= self.num_children():
            return None

        offset = index * self.element_size
        
        row, col = 0, 0
        if self.is_row_major:
            row = index // self.cols
            col = index % self.cols
        else:
            row = index % self.rows
            col = index // self.rows

        child_name = "[%d,%d]" % (row, col)
        if self.rows == 1:
            child_name = "[%d]" % col
        elif self.cols == 1:
            child_name = "[%d]" % row

        return self.valobj.CreateValueFromAddress(child_name, self.data_address + offset, self.element_type)

