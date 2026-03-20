%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.09s
%% Conflict analysis                       : 0.202s
%%   - Positive Examples                   : 0.202s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0s
%% Hypothesis Search                       : 0.051s
%% Total                                   : 0.364s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d1(V1) :- d4(V1).
d2(V1) :- d4(V1).
d3(V1) :- d4(V1).
d0(V1) :- d1(V1); not d2(V1).

