d0(o3).
d1(o0).
d3(o2).
d0(o0).

d2(X) :- not d0(X), d1(X), d3(X).
d3(X) :- not d2(X), d1(X), d0(X).
d2(X) :- not d1(X), d0(X), d3(X).