#constant(obj, o0).
#constant(obj, o2).

#modeh(d3(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).

#pos(eg1, {
  d3(o2)
}, {
  d3(o0)
}, {
  d0(o2).
  d2(o0).
  d2(o2).
  d4(o0).
}).