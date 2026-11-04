%% Uses the Sader-Jarvis algorithm for the F_even deconvolution
%
%  Arguments:
%  ------------
%   zk       - z axis
%   ktscap   - cap-averaged force gradient (measured; from df)
%   A0       - oscillation amplitude (zero-peak)
%   sgwin    - Window size for Savitzky-Golay filter
%   sgdegree - Degree of Savitzky-Golay filter polynom
%   tozero   - shift df data to zero at large z for tozero values 
%              (0: no shift along vertical axis)
%
%  Returns:
%  ------------
%   zFeven   - z axis
%   Feven    - even force
%  
%  ver08, 14.04.2020, Philipp Rahe (prahe@uos.de)
%
%  CHANGELOG:
%    14.04.2020 v08: corrected sign in central diff and abs function
%                    removed in corr0 to corr2
%    03.03.2020 v07: corrected savgol branch, renamed savgol coefficients
%                    removed ktscap filter from savgol branch
%    28.02.2020 v06: added ddf, ktscap to savgol filter branch
%    20.10.2019 v05: changed differentiation to central diff, 
%                    removed variables nd and di
%    14.06.2017 v04: Adapted to Quantitative AFM, added new SG-filters
%    09.11.2014 v03: Corrected line 48: deltaz = (zdf(N)-zdf(1))/(N-1)
%                     (was deltaz = (zdf(N)-zdf(1))/N before)
%    19.09.2013 v02: Added offset subtraction and checks for fields in
%                     used structs
%    27.09.2013 v01: first, tested version (20120924 - 1100am)
function [zFeven, Feven] = Feven_deconv(zk, ktscap, A0, sgwin, sgdegree, tozero)

  % different checks on input parameters
  N = numel(zk);
  if N ~= numel(ktscap)
    error('Lengths of z (len %u) and df (len %u) do not match.', ...
          numel(zk), numel(ktscap));
  end
  if(~iscolumn(zk)), zk = zk';  end
  if(~iscolumn(ktscap)), ktscap = ktscap';  end
  
  % shift data to zero at large z (if requested)
  if(tozero > 0) 
    ktscap = ktscap - mean(ktscap((length(ktscap)-tozero):end));
  end
  
  % calculate the derivation
  if( (sgwin>1) && (sgdegree>=1) && (sgwin>sgdegree) && mod(sgwin,2)>0 ) 
    [~, ddfdz] = savgolfilter(zk, ktscap, sgwin, 1, sgdegree);
    % TODO: Filter ideally improves data, but following line seems to scale
    % the resulting data incorrectly
    %ktscap     = savgolfilter(zk, ktscap, sgwin, 0, sgdegree);
    
    % central differentiation for z (left/right diff for edges)
    ddz = zk.*0;
    ddz(1)       = zk(2)-zk(1);
    ddz(2:end-1) = (zk(3:end) - zk(1:end-2))/2.;
    ddz(end)     = zk(end)-zk(end-1);
    
  else
    if( (sgwin~=0) )
      warning('Invalid filter parameter. Using central differentiation.');
    end
    
    % central differentiation (left/right diff for edges)
    ddfdz = ktscap.*0;
    ddfdz(1)       = (ktscap(2)-ktscap(1))./(zk(2)-zk(1));
    ddfdz(2:end-1) = (ktscap(3:end) - ktscap(1:end-2))./(zk(3:end) - zk(1:end-2));
    ddfdz(end)     = (ktscap(end)-ktscap(end-1))./(zk(end)-zk(end-1));
    
    % central differentiation (left/right diff for edges)
    ddz = zk.*0;
    ddz(1)       = zk(2)-zk(1);
    ddz(2:end-1) = (zk(3:end) - zk(1:end-2))/2.;
    ddz(end)     = zk(end)-zk(end-1);
  end
  
  % prepare output array
  Feven = zeros(1,N-2);
  
  % iterate through data
  for j=1:(N-2)
    
    % integration
    int = -trapz(zk((j+1):N),...
                 (1 + sqrt(A0./(64*pi.*(zk((j+1):N)-zk(j))))).*...
                           ktscap((j+1):N) ...
	               - sqrt(A0.^3./(2.*(zk((j+1):N)-zk(j)))).* ...
                           ddfdz((j+1):N));

    % correction factors to remove pole of the integration
    corr0 = -ktscap(j)*ddz(j);
    corr1 = -sqrt(A0/(16*pi))*ktscap(j)*sqrt(ddz(j));
    corr2 = sqrt(4*A0^3/2).*ddfdz(j).*sqrt(ddz(j));
    
    % resulting even force
    Feven(j) = int + corr0 + corr1 + corr2;
    
  end
  
  % return according z axis. 
  zFeven = zk(1:N-2);


end



