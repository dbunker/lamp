%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.009s
%% Hypothesis Space Generation             : 0.187s
%% Conflict analysis                       : 0.655s
%%   - Positive Examples                   : 0.655s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 0.171s
%% Total                                   : 1.053s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d2(V1) :- d1(V1).
d1(V1) :- d0(V1).
d5(V1) :- d4(V1).
d3(V1) :- d4(V1).
d3(V1) :- d2(V1); not d0(V1).

