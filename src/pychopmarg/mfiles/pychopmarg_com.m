function rslts = pychopmarg_com(com_func, cfg_file, thru_file, fext_files, next_files)
% PYCHOPMARG_COM  Run MATLAB COM on one channel set, for PyChOpMarg (via the MATLAB Engine API for Python).
%
%   rslts = pychopmarg_com(com_func, cfg_file, thru_file, fext_files, next_files)
%
%   com_func:   Name of the MATLAB COM function (e.g. 'com_ieee8023_480'); must be on the MATLAB path.
%   cfg_file:   COM configuration spreadsheet (*.xls/*.xlsx).
%   thru_file:  Touchstone file of the victim (thru) channel.
%   fext_files: Cell array of far-end aggressor Touchstone files (may be empty).
%   next_files: Cell array of near-end aggressor Touchstone files (may be empty).
%
%   rslts: Cell array, one scalar struct per COM package test case, holding each field of COM's own
%          results (`output_args`) that the Engine API can return to Python: numeric/logical arrays and
%          character vectors. (Nested structs, struct arrays, cells, etc. are omitted.)
%
% Calling COM with an output argument makes it return its results, instead of printing them.

fext_files = as_cell(fext_files);
next_files = as_cell(next_files);
args = [{cfg_file, numel(fext_files), numel(next_files), thru_file}, fext_files(:)', next_files(:)'];
results = feval(com_func, args{:});
if ~iscell(results)
    results = {results};
end
rslts = cell(1, numel(results));
for k = 1:numel(results)
    rslts{k} = returnable_fields(results{k});
end
end

function c = as_cell(x)
% Python lists arrive as cell arrays; an empty one may arrive as `[]`.
if iscell(x)
    c = x;
elseif isempty(x)
    c = {};
else
    c = {x};
end
end

function s = returnable_fields(r)
s = struct();
if ~isstruct(r) || ~isscalar(r)
    return
end
names = fieldnames(r);
for i = 1:numel(names)
    v = r.(names{i});
    if (isnumeric(v) || islogical(v)) && ismatrix(v)
        s.(names{i}) = double(v);
    elseif ischar(v) && (isrow(v) || isempty(v))
        s.(names{i}) = v;
    end
end
end
