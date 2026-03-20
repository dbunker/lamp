%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 0.077s
%% Conflict analysis                       : 0.137s
%%   - Positive Examples                   : 0.137s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0s
%% Hypothesis Search                       : 0.031s
%% Total                                   : 0.26s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d3(V1) :- d5(V1).
d1(V1) :- d5(V1); not d4(V1).
d2(V1) :- d4(V1); not d5(V1).

