d0(o3).
d1(o4).
d3(o4).
d0(o0).
d4(o0).

d3(X) :- not d0(X), not d1(X), d2(X).
d2(X) :- not d3(X), d0(X), not d4(X).