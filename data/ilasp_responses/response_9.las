%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.311s
%% Conflict analysis                       : 0.848s
%%   - Positive Examples                   : 0.848s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 0.226s
%% Total                                   : 1.443s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d3(V1) :- d1(V1).
d4(V1) :- d1(V1).
d0(V1) :- d4(V1); d8(V1).

