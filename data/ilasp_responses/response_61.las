%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 0.191s
%% Conflict analysis                       : 0.56s
%%   - Positive Examples                   : 0.56s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.001s
%% Hypothesis Search                       : 0.129s
%% Total                                   : 0.914s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d3(V1) :- d5(V1).
d2(V1) :- d0(V1).
d1(V1) :- d2(V1); not d0(V1).
d4(V1) :- d2(V1); not d5(V1).

