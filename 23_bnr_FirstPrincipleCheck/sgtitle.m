function ht = sgtitle(arg1, arg2, varargin)
%SGTITLE Create a title over a grid of subplots (MATLAB compatible)
%
%   sgtitle(str) creates a centered title at the top of the current figure.
%
%   sgtitle(fig, str) creates a title for the specified figure.
%
%   ht = sgtitle(...) returns the text object handle.
%
%   sgtitle(..., 'Name', Value, ...) sets text properties such as
%   'FontSize', 'FontWeight', 'Color', 'Interpreter', etc.
%
%   This is a compatibility implementation for GNU Octave.

  if nargin < 1
    error('sgtitle:NotEnoughInputs', 'Not enough input arguments.');
  end

  % Parse inputs: sgtitle(str, ...) or sgtitle(fig, str, ...)
  if ishghandle(arg1) && strcmp(get(arg1, 'Type'), 'figure')
    fig = arg1;
    if nargin < 2
      error('sgtitle:NotEnoughInputs', ...
            'You must provide a title string when passing a figure handle.');
    end
    titleStr = arg2;
    props = varargin;
  else
    fig = gcf;
    titleStr = arg1;
    if nargin > 1
      props = [{arg2} varargin];
    else
      props = {};
    end
  end

  if ~ishghandle(fig)
    fig = figure;
  end

  % Tag used to identify and replace previous sgtitle in the same figure
  tagTitle = 'OctaveSgtitle';
  tagAxes  = 'OctaveSgtitleAxes';

  % Delete any existing figure level title created by this function
  oldTitle = findall(fig, 'Type', 'text', 'Tag', tagTitle);
  if ~isempty(oldTitle)
    delete(oldTitle);
  end
  oldAxes = findall(fig, 'Type', 'axes', 'Tag', tagAxes);
  if ~isempty(oldAxes)
    delete(oldAxes);
  end

  % Save current axes so we can restore it after creating overlay axes
  oldCurrentAxes = get(fig, 'CurrentAxes');

  % Create an invisible, normalized axes that spans the whole figure
  ax = axes('Parent', fig, ...
            'Units', 'normalized', ...
            'Position', [0 0 1 1], ...
            'Visible', 'off', ...
            'Tag', tagAxes, ...
            'HitTest', 'off');

  % Put the new axes on top if uistack exists
  if exist('uistack', 'file') || exist('uistack', 'builtin')
    try
      uistack(ax, 'top');
    catch
      % If something fails, silently ignore
    end
  end

  % Create the title text near the top center of the figure
  ht = text(ax, 0.5, 0.98, titleStr, ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', ...
            'Tag', tagTitle);

  % Apply any Name-Value pairs to the text object
  if ~isempty(props)
    try
      set(ht, props{:});
    catch err
      warning('sgtitle:InvalidProperty', ...
              'Some properties could not be applied: %s', err.message);
    end
  end

  % Restore previous current axes if it still exists
  if ishghandle(oldCurrentAxes)
    set(fig, 'CurrentAxes', oldCurrentAxes);
  end

  % If no output is requested, do not show the handle
  if nargout == 0
    clear ht;
  end
end

