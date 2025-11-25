d0(o3).
d3(o4).
d4(o0).
d2(o4).

d1(X) :- not d2(X), d0(X), d4(X).
d4(X) :- d2(X), d0(X), not d3(X).
d4(X) :- not d1(X), d2(X), d3(X).