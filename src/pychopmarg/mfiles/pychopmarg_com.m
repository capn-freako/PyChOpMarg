function rslts = pychopmarg_com(com_func, cfg_file, thru_file, fext_files, next_files, pulse_resps)
% PYCHOPMARG_COM  Run MATLAB COM on one channel set, for PyChOpMarg (via the MATLAB Engine API for Python).
%
%   rslts = pychopmarg_com(com_func, cfg_file, thru_file, fext_files, next_files [, pulse_resps])
%
%   com_func:    Name of the MATLAB COM function (e.g. 'com_ieee8023_4p17p0'); must be on the MATLAB path.
%   cfg_file:    COM configuration spreadsheet (*.xls/*.xlsx).
%   thru_file:   Touchstone file of the victim (thru) channel.
%   fext_files:  Cell array of far-end aggressor Touchstone files (may be empty).
%   next_files:  Cell array of near-end aggressor Touchstone files (may be empty).
%   pulse_resps: Also return the victim's pulse responses, when true. (Default: false.)
%
%   rslts: Cell array, one scalar struct per COM package test case, holding each field of COM's own
%          results (`output_args`) that the Engine API can return to Python: numeric/logical arrays and
%          character vectors. (Nested structs, struct arrays, cells, etc. are omitted.)
%          With `pulse_resps`, each also holds the victim's (`chdata(1)`) time vector and unequalized and
%          fully equalized pulse responses: `vic_t`, `vic_uneq_pulse_response`, and `vic_eq_pulse_response`.
%
% Calling COM with an output argument makes it return its results, instead of printing them.
% COM includes `chdata` in its results only with its `BREAD_CRUMBS` option, so `pulse_resps` runs COM on a
% temporary copy of the configuration with that option set. (COM reads a *.mat configuration -- the sheet's
% contents, as `parameter` -- but mangles any folder in its path, so the copy goes in the working folder.)

if nargin < 6
    pulse_resps = false;
end
fext_files = as_cell(fext_files);
next_files = as_cell(next_files);
cfg = cfg_file;
if pulse_resps
    cfg = bread_crumbs_config(cfg_file);
    cleanup = onCleanup(@() delete(cfg)); %#ok<NASGU>
end
args = [{cfg, numel(fext_files), numel(next_files), thru_file}, fext_files(:)', next_files(:)'];
results = feval(com_func, args{:});
if ~iscell(results)
    results = {results};
end
rslts = cell(1, numel(results));
for k = 1:numel(results)
    rslts{k} = returnable_fields(results{k});
    if pulse_resps && isfield(results{k}, 'chdata')
        vic = results{k}.chdata(1);
        rslts{k}.vic_t = double(vic.t(:)');
        rslts{k}.vic_uneq_pulse_response = double(vic.uneq_pulse_response(:)');
        rslts{k}.vic_eq_pulse_response = double(vic.eq_pulse_response(:)');
    end
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

function name = bread_crumbs_config(cfg_file)
% A copy of the configuration in `cfg_file`, with `BREAD_CRUMBS` set, as a *.mat file in the working folder.
try  % (Read as COM itself does.)
    [~, ~, parameter] = xlsread(cfg_file, 'COM_Settings', '', '');  %#ok<XLSRD>
catch
    [~, ~, parameter] = xlsread(cfg_file, 'COM_Settings', '', 'basic');  %#ok<XLSRD>
end
[row, col] = find(strcmpi(parameter, 'BREAD_CRUMBS'));
if isempty(row)
    parameter(end + 1, 1:2) = {'BREAD_CRUMBS', 1};
else
    parameter{row(1), col(1) + 1} = 1;
end
name = sprintf('pychopmarg_com_cfg_%d.mat', feature('getpid'));
save(name, 'parameter');
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
