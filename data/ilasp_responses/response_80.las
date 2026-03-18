%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.078s
%% Conflict analysis                       : 0.148s
%%   - Positive Examples                   : 0.148s
%% Counterexample search                   : 0.003s
%%   - CDOEs                               : 0.001s
%%   - CDPIs                               : 0.001s
%% Hypothesis Search                       : 0.041s
%% Total                                   : 0.287s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d4(V1) :- d3(V1).
d2(V1) :- d4(V1).
d0(V1) :- d5(V1).
d2(V1) :- d5(V1).

