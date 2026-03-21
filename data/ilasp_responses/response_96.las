%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.185s
%% Conflict analysis                       : 0.633s
%%   - Positive Examples                   : 0.633s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0s
%% Hypothesis Search                       : 0.175s
%% Total                                   : 1.033s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d2(V1) :- d1(V1).
d1(V1) :- d0(V1).
d5(V1) :- d4(V1).
d3(V1) :- d4(V1).
d3(V1) :- d2(V1); not d0(V1).

