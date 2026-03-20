%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 0.198s
%% Conflict analysis                       : 0.969s
%%   - Positive Examples                   : 0.969s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.001s
%% Hypothesis Search                       : 0.321s
%% Total                                   : 1.54s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d5(V1) :- d1(V1).
d3(V1) :- d2(V1).
d0(V1) :- d4(V1).
d1(V1) :- d0(V1).
d4(V1) :- d5(V1); not d2(V1).

